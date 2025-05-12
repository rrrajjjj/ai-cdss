from typing import List

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame
from scipy import signal

from ai_cdss.models import ScoringSchema, PPFSchema, SessionSchema, TimeseriesSchema
from ai_cdss.processing import safe_merge, apply_savgol_filter_groupwise, get_rolling_theilsen_slope
from ai_cdss.constants import (
    BY_PP, BY_PPS, BY_PPST, 
    PROTOCOL_ID, 
    SESSION_ID,
    DM_KEY, PE_KEY,
    DM_VALUE, PE_VALUE,
    ADHERENCE, 
    USAGE,
    DAYS,
    WEEKDAY_INDEX,
    PRESCRIPTION_ENDING_DATE,
    PRESCRIPTION_ACTIVE,
    FINAL_METRICS, 
    SAVGOL_WINDOW_SIZE,
    SAVGOL_POLY_ORDER,
    THEILSON_REGRESSION_WINDOW_SIZE
)
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Data Processor Class

class DataProcessor:
    """
    A class for processing patient session data, applying Exponential Weighted 
    Moving Average (EWMA) and computing a final weighted score.

    The final score is computed as:

    .. math::

        S = \\alpha \\cdot A + \\beta \\cdot DM + \\gamma \\cdot PPF

    where:

    - :math:`A` is Adherence
    - :math:`DM` is the Difficulty Modulator
    - :math:`PPF` is the Patient Prescription Factor

    Parameters
    ----------
    weights : List[float]
        List of weights :math:`\\alpha`, :math:`\\beta`, :math:`\\gamma` for computing the final score.
    alpha : float
        The smoothing factor for EWMA, controlling how much past values influence the trend.
    """
    def __init__(
        self,
        weights: List[float] = [1,1,1], 
        alpha: float = 0.5
    ):
        """
        Initialize the data processor with optional weights for scoring.
        """
        self.weights = weights
        self.alpha = alpha

    @pa.check_types
    def process_data(
        self, 
        session_data: DataFrame[SessionSchema], 
        timeseries_data: DataFrame[TimeseriesSchema], 
        ppf_data: DataFrame[PPFSchema], 
        init_data: pd.DataFrame
    ) -> DataFrame[ScoringSchema]:
        """
        Process and score patient-protocol combinations using session, timeseries, and PPF data.

        Applies EWMA to adherence and difficulty modulators, merges features,
        and computes final patient-protocol scores.

        Parameters
        ----------
        session_data : DataFrame[SessionSchema]
            Session-level data including adherence and scheduling information.
        timeseries_data : DataFrame[TimeseriesSchema]
            Timepoint-level data including DMs and performance metrics.
        ppf_data : DataFrame[PPFSchema]
            Patient-protocol fitness values and contributions.

        Returns
        -------
        DataFrame[ScoringSchema]
            Final scored dataframe with protocol recommendations.
        """

        # 1. Preprocess time series
        ts_processed = self.preprocess_timeseries(timeseries_data)
        # 2. Preprocess session data
        ss_processed = self.preprocess_sessions(session_data)
        # 3.1 Merge session and timeseries data
        merged_data = safe_merge(ss_processed, ts_processed, on=BY_PPS, how="inner", left_name="session", right_name="ts")
        # 3.2 Aggregate metrics per protocol
        data = merged_data.groupby(by=BY_PP)[FINAL_METRICS].last()
        # 3.3 Merge session, timeseries, and ppf
        data = ppf_data.merge(data, on=BY_PP, how="left")
        # 4. Compute scores
        scored_data = self.compute_score(data, init_data)
        # 5. Propagate metadata
        scored_data.attrs = ppf_data.attrs

        return scored_data
    
    def preprocess_timeseries(self, timeseries_data: pd.DataFrame) -> pd.DataFrame:
        timeseries_data = timeseries_data.groupby(BY_PPS).agg({DM_VALUE:"mean", PE_VALUE:"mean"}).reset_index()
        timeseries_data['DM_SMOOTH'] = timeseries_data.groupby(by=BY_PP)[DM_VALUE].transform(apply_savgol_filter_groupwise, SAVGOL_WINDOW_SIZE, SAVGOL_POLY_ORDER)
        timeseries_data['DM_VALUE'] = timeseries_data.groupby(by=BY_PP, group_keys=False).apply(
            lambda g: get_rolling_theilsen_slope(g['DM_SMOOTH'], THEILSON_REGRESSION_WINDOW_SIZE)
        ).fillna(0)
    
        # Drop DM_SMOOTH column
        timeseries_data = timeseries_data.drop(columns='DM_SMOOTH')
        ts = timeseries_data.sort_values(by=BY_PPS)
        
        return ts

    def preprocess_sessions(self, session_data: pd.DataFrame) -> pd.DataFrame:
        session = session_data.sort_values(by=BY_PPS)
        
        # Compute ewma
        session = self._compute_ewma(session, ADHERENCE, BY_PP)

        # Compute usage
        session[USAGE] = session.groupby(BY_PP)[SESSION_ID].transform("nunique").astype("Int64")
        
        # Compute prescription days
        # TODO: Improve to get prescriptions days for all weeks not just last prescriptions
        prescribed_days = (
            session[session[PRESCRIPTION_ENDING_DATE] == PRESCRIPTION_ACTIVE]
            .groupby(BY_PP)[WEEKDAY_INDEX]
            .agg(lambda x: sorted(x.unique()))
            .rename(DAYS)
        )

        # Merge days into session DataFrame
        session = session.merge(prescribed_days, on=BY_PP, how="left")

        return session

    def compute_score(self, data, protocol_metrics):
        """
        Initializes metrics based on legacy data and computes score

        Parameters
        ----------
        data : pd.DataFrame
            Aggregated session and timeseries data per patient-protocol.

        Returns
        -------
        pd.DataFrame
            Scored DataFrame sorted by patient and protocol.
        """
        # Initialize missing values
        data = self._init_metrics(data, protocol_metrics)
        
        # Compute objective function score alpha*Adherence + beta*DM + gamma*PPF
        score = self._compute_score(data)
        
        # Sort the output dataframe
        score.sort_values(by=BY_PP, inplace=True)

        return score

    def aggregate_dms_by_time(self, timeseries_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate multiple difficulty modulator (DM) values at the same timepoint.

        Groups by patient, protocol, session, and time, and computes average DM and PE.

        Parameters
        ----------
        timeseries_data : pd.DataFrame
            Raw timeseries data with DM_KEY, DM_VALUE, PE_KEY, PE_VALUE.

        Returns
        -------
        pd.DataFrame
            Aggregated timeseries data with unique timepoints.
        """
        return (
            timeseries_data
            .sort_values(by=BY_PPST) # Sort df by time
            .groupby(BY_PPST)
            .agg({
                DM_KEY: lambda x: tuple(set(x)),  # Unique parameters at this time
                DM_VALUE: "mean",               # Average parameter value
                PE_KEY: "first",                # Assume same performance key per time, take first
                PE_VALUE: "mean"                # Average performance value (usually only one)
            })
            .reset_index()
        )

    def _init_metrics(self, data: pd.DataFrame, protocol_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing protocol-level metrics with zeros.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with potentially missing features.

        Returns
        -------
        pd.DataFrame
            Safe dataframe with no NaNs.
        """
        # data = data.fillna(0)

        # Only fill NaNs using map for matching PROTOCOL_IDs
        data[ADHERENCE] = data[ADHERENCE].fillna(data[PROTOCOL_ID].map(protocol_metrics[ADHERENCE]))
        data[DM_VALUE] = data[DM_VALUE].fillna(data[PROTOCOL_ID].map(protocol_metrics["DM_DELTA"]))
        data[PE_VALUE] = data[PE_VALUE].fillna(0)
        data[DAYS]  = data[DAYS].fillna(0)
        data[USAGE] = data[USAGE].fillna(0)
        data = data.fillna(0)
        return data
    
    def _compute_ewma(self, df, value_col, group_cols, sufix=""):
        """
        Compute Exponential Weighted Moving Average (EWMA) over grouped time series.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing time-series values.
        value_col : str
            The column for which EWMA should be calculated.
        group_cols : list of str
            Columns defining the group over which to apply the EWMA.

        Returns
        -------
        pd.DataFrame
            DataFrame with EWMA column replacing the original.
        """
        return df.assign(
            **{f"{value_col}{sufix}": df.groupby(by=group_cols)[value_col].transform(lambda x: x.ewm(alpha=self.alpha, adjust=True).mean())}
        )

    def _compute_score(self, scoring: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the final score based on adherence, DM, and PPF values.

        Applies the weighted scoring formula:

        .. math::

            S = \\alpha \\cdot A + \\beta \\cdot DM + \\gamma \\cdot PPF

        Parameters
        ----------
        scoring : pd.DataFrame
            DataFrame containing columns: ADHERENCE, DM_VALUE, PPF.

        Returns
        -------
        pd.DataFrame
            DataFrame with an added SCORE column.
        """
        scoring_df = scoring.copy()
        scoring_df['SCORE'] = (
            scoring_df['ADHERENCE'] * self.weights[0]
            + scoring_df['DM_VALUE'] * self.weights[1]
            + scoring_df['PPF'] * self.weights[2]
        )
        return scoring_df
