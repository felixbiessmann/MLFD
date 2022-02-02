import numpy as np
import pandas as pd
import uuid
from sklearn.metrics import classification_report

class Inspector:
    def __init__(self,
                 assume_errors_known: bool = True):
        self.assume_errors_known = assume_errors_known

        self._error_positions = None
        self._predicted_error_positions = None
        self._cleaning_error_positions = None

    def cleaning_report(self):
        """
        A full report on the cleaning task, inspired by sklearn's
        classification_report.
        """
        pass

    def compare_series(self, se_a, se_b):
        """
        Get a boolean selector that selects differences in two series.

        We are careful working with missing values, as NaN == NaN resolves to
        False.
        """
        # This makes comparison operations work for missing values.
        fill = str(uuid.uuid4())
        return se_a.fillna(fill) != se_b.fillna(fill)

    def calculate_error_positions(self, y_clean: pd.Series, y_dirty: pd.Series):
        """
        Calculates error positions in the dirty series by comparing the dirty
        series to a series containing the ground truth (y_clean).
        """
        self._error_positions = self.compare_series(y_clean, y_dirty)


    def calculate_cleaning_error_positions(self,
                                           y_clean: pd.Series,
                                           y_pred: pd.Series):
        """
        Calculates error positions in the predicted series by comparing the
        predicted series to a series containing the ground truth (y_clean).
        """
        self._cleaning_error_positions = self.compare_series(y_clean, y_pred)

    def inspect_cleaning_results(self,
                                 df_clean: pd.DataFrame,
                                 df_pred: pd.DataFrame,
                                 df_dirty: pd.DataFrame,
                                 context_selector,
                                 context_height: int = 3):

        error_indices = self._cleaning_error_positions.index[self._cleaning_error_positions == True]
        for i, pos in enumerate(error_indices):
            row_start, row_end = pos-context_height, pos+context_height
            print(f"Evaluating error {i} from {len(self._cleaning_error_positions)}")
            print(f"Error in row {pos}:")
            print(df_dirty.iloc[row_start:row_end, :].loc[:, context_selector])
            print(f"Cleaning result in row {pos}:")
            print(df_pred.iloc[row_start:row_end, :].loc[:, context_selector])
            print(f"Groud truth in row {pos}:")
            print(df_clean.iloc[row_start:row_end, :].loc[:, context_selector])



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

        Most data-cleaning publications work under the assumption that all
        errors have been successfully detected. (Mahdavi 2020) This behavior
        can be controlled with the parameter assume_errors_known. If we work
        under this assumptions, true negatives and false positives become
        impossible.
        """
        if self.assume_errors_known:
            y_clean = y_clean.loc[self._error_positions]
            y_pred = y_pred.loc[self._error_positions]
            y_dirty = y_dirty.loc[self._error_positions]

        tp = sum(np.logical_and(y_dirty != y_clean, y_pred == y_clean))
        fp = sum(np.logical_and(y_dirty == y_clean, y_pred != y_clean))
        fn = sum(np.logical_and(y_dirty != y_clean, y_pred != y_clean))
        tn = sum(np.logical_and(y_dirty == y_clean, y_pred == y_clean))

        print("Calculating Cleaning Performance.")
        print(f"Counted {tp} TPs, {fp} FPs, {fn} FNs and {tn} TNs.")

        p = .0 if (tp + fp) == 0 else tp / (tp + fp)
        r = .0 if (tp + fn) == 0 else tp / (tp + fn)
        f1_score = .0 if (p+r) == 0 else 2 * (p*r)/(p+r)
        return f1_score


    def error_detection_performance(self,
                                    y_pred: pd.Series,
                                    y_dirty: pd.Series):
        """
        Calculate the f1-score for finding the correct position of errors in
        y_dirty.
        """
        self._predicted_error_positions = y_dirty != y_pred

        report = classification_report(self._error_positions, self._predicted_error_positions)
        return report
