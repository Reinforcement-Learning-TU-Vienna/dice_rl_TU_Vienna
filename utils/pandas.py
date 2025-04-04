# ------------------------------------------------ #

def head_by_id(df, n=1, id_name="id"):
    ids_unique = df[id_name].unique()

    if isinstance(n, int):
        ids_select = ids_unique[:n]
    elif isinstance(n, list):
        ids_select = ids_unique[n]
    else:
        raise ValueError

    f = df[id_name].isin(ids_select)
    return df[f]

# ------------------------------------------------ #
