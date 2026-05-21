# Data Cleaning & Feature Engineering Operations

## Data Cleaning

1. **Drop the `Cabin` column** — Removed from both train and test sets due to a high number of null values.
2. **Impute missing `Embarked` values** — Filled null values in the `Embarked` column with `'S'` (most frequent port).
3. **Impute missing `Fare` values** — Filled null values in the test set's `Fare` column with the column mean.
4. **Impute missing `Age` values** — Filled null ages with the median `Age` grouped by `Sex` and `Pclass`.
5. **Convert `Age` to integer** — Cast the `Age` column from float to `int64`.

## Feature Engineering

6. **Extract titles from `Name`** — Parsed the `Name` column to extract the passenger title (e.g., Mr, Mrs, Miss) into a new `Title` column.
7. **Consolidate rare titles** — Grouped infrequent titles (`Lady`, `Countess`, `Capt`, `Col`, `Don`, `Dr`, `Major`, `Rev`, `Sir`, `Jonkheer`, `Dona`) into a single `'Rare'` category. Normalized `Mlle` → `Miss`, `Ms` → `Miss`, and `Mme` → `Mrs`.
8. **Create `Family_size` column** — Combined `SibSp + Parch + 1` into a new numeric `Family_size` feature.
9. **Drop redundant columns** — Removed `Name`, `Parch`, `SibSp`, and `Ticket` columns after feature extraction.
10. **Categorize `Family_size`** — Converted the numeric `Family_size` into categorical bins: `Alone` (1), `Small` (2–4), and `Large` (5+).
