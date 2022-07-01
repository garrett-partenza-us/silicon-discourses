def padding(arr, seq_len, mode="right"):
    assert mode=="right" or mode=="left", "invalid padding mode"
    if mode=="right":
        return F.pad(arr, (0, seq_len-arr.nelement()))
    else:
        return F.pad(arr, (seq_len-arr.nelement()), 0)