

def prepare_dataset(
    file_dir: str,
    periods: int = 1,
    order: int = 1
) -> pd.DataFrame:
    df = load_dataset(file_dir)
    prepared_df = differencing(df, periods=periods, order=order)
    prepared_df.head()
    prepared_df.dropna(inplace=True)

    print("First few rows of dataset loaded:")
    print(prepared_df.head())
    return prepared_df


# Normalize the sequence
def normalize(
    df: pd.DataFrame,
    train_ratio: float,
    lags: int
) -> Tuple[np.ndarray]:
    scaler = StandardScaler().fit(
        df[:int(train_ratio*len(df))].values)
    df.iloc[:, 0] = scaler.transform(df.values)

    X_raw, y_raw = gen_supervised_sequence(
        df, lags, df.columns[0], sequential_label=False)

    (X_train, X_test, y_train, y_test) = train_test_split(
        X_raw, y_raw,
        test_size=1 - train_ratio,
        shuffle=False)

    (X_train, X_val, y_train, y_val) = train_test_split(
        X_train, y_train,
        test_size=0.1,
        shuffle=False
    )

    def trans(x): return x.reshape(-1, 1)

    y_train = trans(y_train)
    y_test = trans(y_test)
    y_val = trans(y_val)

    print(
        f"Training and testing set generated,\
        \nX_train shape: {X_train.shape}\
        \ny_train shape: {y_train.shape}\
        \nX_test shape: {X_test.shape}\
        \ny_test shape: {y_test.shape}\
        \nX_validation shape: {X_val.shape}\
        \ny_validation shape: {y_val.shape}")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )
