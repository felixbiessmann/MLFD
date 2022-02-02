import numpy as np
import pandas as pd
import uuid

class Inspector:
    def __init__(self,
                 assume_errors_known: bool = True):
        self.assume_errors_known = assume_errors_known

    def cleaning_performance(self,
                             y_clean: pd.Series,
                             y_pred: pd.Series,
                             y_dirty: pd.Series):
        """
        Calculate the f1-score between the clean labels and the predicted
        labels.

        As defined by Rekasinas et al. 2017 (Holoclean), we compute:
        - Precision as the fraction of correct repairs over the total number
          of repairs performed.
        - Recall as the fraction of (correct repairs of real errors) over the
          total number of errors.

        Also, most data-cleaning publications work under the assumption that all
        errors have been successfully detected. (see Mahdavi 2020)

        Be careful working with missing values, as NaN == NaN resolves to
        False.

        TODO: Return classification report instead of just f1_score.
        """
        # This makes comparison operations work for missing values.
        fill = str(uuid.uuid4())
        y_clean.fillna(fill, inplace=True)
        y_dirty.fillna(fill, inplace=True)

        error_positions = y_clean != y_dirty
        if self.assume_errors_known:
            y_clean = y_clean.loc[error_positions]
            y_pred = y_pred.loc[error_positions]
            y_dirty = y_dirty.loc[error_positions]

        tp = sum(np.logical_and(y_dirty != y_clean, y_pred == y_clean))
        fp = sum(np.logical_and(y_dirty == y_clean, y_pred != y_clean))
        fn = sum(np.logical_and(y_dirty != y_clean, y_pred != y_clean))
        tn = sum(np.logical_and(y_dirty == y_clean, y_pred == y_clean))

        logger.debug("Calculating Cleaning Performance.")
        logger.debug(f"Counted {tp} TPs, {fp} FPs, {fn} FNs and {tn} TNs.")

        p = 0 if (tp + fp) == 0 else tp / (tp + fp)
        r = 0 if (tp + fn) == 0 else tp / (tp + fn)
        f1_score = 0 if (p+r) == 0 else 2 * (p*r)/(p+r)
        return f1_score


    def error_detection_performance(self,
                                    y_clean: pd.Series,
                                    y_pred: pd.Series,
                                    y_dirty: pd.Series):
        """
        Calculate the f1-score for finding the correct position of errors in
        y_dirty.

        TODO: Return classification report instead of just f1_score.
        """
        logger = logging.getLogger('pfd')
        y_error_position_true = y_clean != y_dirty
        y_error_position_pred = y_dirty != y_pred
        rep = classification_report(y_error_position_true, y_error_position_pred)
        logger.debug("Calculating Error Detection Performance")
        logger.debug(f'Counted {sum(y_error_position_true)} errors in the original data.')
        logger.debug(f'And {sum(y_error_position_pred)} errors were predicted.')
        return f1_score(y_error_position_true, y_error_position_pred)

