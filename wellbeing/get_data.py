from typing import Literal, Union  # , Tuple, Any

# import pickle
import dill as pickle
import requests
import os
from urllib.parse import urlparse
import hashlib
import json

import pandas as pd
from fredapi import Fred

import beaapi

# ^^^ Once we upgrade to Python >= 3.9, can use importlib.resources

cache_dir: str = "cache"


# Small update to dotenv.dotenv_values
def dotenv_values_lcl(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return dict(line.split("=") for line in f.read().splitlines() if not line.startswith("#"))


dotenv_values = dotenv_values_lcl
api_keys = dotenv_values("apiKeys.env")


def recessions(frequency: Literal["A", "Q"], from_cache=False, save_to_cache=True) -> pd.DataFrame:
    """Gives you consolidated ranges of just recessions (with stop and end datetimes).
    Uses monthly recession indicator and then aggregates to Quarter or Year.
    Somtimes the NBER Quarter turning point doesn't contain the Monthly turning point.
    See (https://www.nber.org/research/data/us-business-cycle-expansions-and-contractions),
    but since we need this for quarters and years, we will aggregate from the same base
    series.

    Args:
        frequency (Literal['A' 'Q']): _description_
        from_cache (bool, optional): _description_. Defaults to False.
        save_to_cache (bool, optional): _description_. Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """
    series = "USREC"
    start = "1929-01-01"
    cache_fname = "fred_" + series + "_" + start + ".pkl"
    if not from_cache:
        if "fred_key" not in api_keys:
            raise Exception("Couldn't find a fred_key entry in apiKeys.env")
        fred_data = Fred(api_keys["fred_key"]).get_series(series_id=series, observation_start=start)
        if save_to_cache:
            with open(cache_dir + "/" + cache_fname, "wb") as handle:
                pickle.dump(fred_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_dir + "/" + cache_fname, "rb") as handle:
            fred_data = pickle.load(handle)

    recessions = fred_data.astype(bool)
    recessions.name = "recessions"
    recessions.index = recessions.index.to_period(freq=frequency).to_timestamp(how="start")
    recessions = recessions.groupby(recessions.index).aggregate("any")
    rec_period = (recessions.diff() == True).cumsum()
    recessions = (
        pd.DataFrame({"recessions": recessions, "rec_period": rec_period})
        .reset_index()
        .groupby("rec_period")
        .agg(["min", "max"])
    )
    recessions.columns = ["start", "end", "recession", "recession_copy"]
    recessions = recessions[recessions["recession"]].drop(["recession", "recession_copy"], axis=1)
    recessions["stop"] = recessions["end"] + pd.DateOffset(months=3 if frequency == "Q" else 12)
    return recessions


def business_cycles(from_cache=False, save_to_cache=True) -> pd.Series:
    """_summary_

    Args:
        url (str): _description_
        peaks (dict): _description_
        troughs (dict): _description_
        from_cache (bool, optional): _description_. Defaults to False.
        save_to_cache (bool, optional): _description_. Defaults to True.

    Returns:
        pd.Series: Returns a Pandas Series with a quarterly PeriodIndex
    and values of 1 for peak periods and 0 for trough periods.
    """
    fred_data = fred_series(
        "USRECQ", "1854-12-01", "Q", from_cache=from_cache, save_to_cache=save_to_cache
    )
    fred_data = fred_data.astype(bool)
    fred_data = fred_data == False  # switch around
    fred_data = fred_data[fred_data != fred_data.shift(-1)]  # last one before it changes
    fred_data = fred_data[: len(fred_data) - 1]  # can't tell for final quarter
    return fred_data


def nipa_series(
    table: str,
    frequency: Literal["A", "Q"],
    line: Union[int, str],
    from_cache=False,
    save_to_cache=True,
) -> pd.Series:
    cache_fname = f"NIPA_{table}_{frequency}.pkl"
    if not from_cache:
        if "bea_key" not in api_keys:
            raise Exception("Couldn't find a bea_key entry in apiKeys.env")
        data = beaapi.get_data(
            userid=api_keys["bea_key"],
            datasetname="NIPA",
            TableName=table,
            Frequency=frequency,
            Year="X",
        )
        data.attrs.pop("params")
        data.attrs.pop("detail")
        if save_to_cache:
            with open(cache_dir + "/" + cache_fname, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_dir + "/" + cache_fname, "rb") as handle:
            data = pickle.load(handle)

    nipa_series: pd.Series = data.query(f"LineNumber == {line}").set_index("TimePeriod").DataValue
    nipa_series.index = pd.to_datetime(nipa_series.index).to_period(freq=frequency)
    return nipa_series


def regional_data(
    table: str,
    frequency: Literal["A", "Q"],
    geo: str,
    line: Union[int, str],
    from_cache=False,
    save_to_cache=True,
) -> pd.DataFrame:
    cache_fname = f"Regional_{table}_{frequency}_{geo}_{line}.pkl"
    if not from_cache:
        if "bea_key" not in api_keys:
            raise Exception("Couldn't find a bea_key entry in apiKeys.env")
        data = beaapi.get_data(
            userid=api_keys["bea_key"],
            datasetname="Regional",
            TableName=table,
            Frequency=frequency,
            GeoFips=geo,
            LineCode=line,
            Year="ALL",
        )
        if save_to_cache:
            with open(cache_dir + "/" + cache_fname, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_dir + "/" + cache_fname, "rb") as handle:
            data = pickle.load(handle)
    regional_df: pd.DataFrame = data.set_index("TimePeriod")
    regional_df.index = pd.to_datetime(regional_df.index).to_period(freq=frequency)
    return regional_df


def industry_data(
    table: str, frequency: Literal["A", "Q"], from_cache=False, save_to_cache=True
) -> pd.DataFrame:
    quarters = {"I": "1", "II": "2", "III": "3", "IV": "4"}
    cache_fname = f"Industry_{table}_{frequency}.pkl"
    if not from_cache:
        if "bea_key" not in api_keys:
            raise Exception("Couldn't find a bea_key entry in apiKeys.env")
        data = beaapi.get_data(
            userid=api_keys["bea_key"],
            datasetname="GDPbyIndustry",
            TableID=table,
            Frequency=frequency,
            Industry="ALL",
            Year="ALL",
        )
        if save_to_cache:
            with open(cache_dir + "/" + cache_fname, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_dir + "/" + cache_fname, "rb") as handle:
            data = pickle.load(handle)
    industry_data = (
        data.assign(
            TimePeriod=lambda x: x.Year.astype(str).str.cat(x.Quarter.replace(quarters), sep="Q")
        )
        .set_index("TimePeriod")
        .rename(columns={"IndustrYDescription": "IndustryDescription"})[
            ["Industry", "IndustryDescription", "DataValue"]
        ]
    )
    industry_data.index = pd.to_datetime(industry_data.index).to_period(freq=frequency)
    return industry_data


def mne_series(
    direction: str,
    classification: str,
    nonbank: int,
    ownership: int,
    country: str,
    series_id: str,
    year: Union[str, int, pd.Period] = None,
    from_cache=False,
    save_to_cache=True,
) -> pd.Series:
    cache_fname = (
        f"MNE_{direction}_{classification}_{nonbank}_{ownership}_{country}_{series_id}_{year}.pkl"
    )

    if not from_cache:
        if "bea_key" not in api_keys:
            raise Exception("Couldn't find a bea_key entry in apiKeys.env")
        data = beaapi.get_data(
            userid=api_keys["bea_key"],
            datasetname="MNE",
            DirectionOfInvestment=direction,
            Classification=classification,
            NonbankAffiliatesOnly=nonbank,
            OwnershipLevel=ownership,
            Country=country,
            SeriesID=series_id,
            Year=year,
        )
        if save_to_cache:
            with open(cache_dir + "/" + cache_fname, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_dir + "/" + cache_fname, "rb") as handle:
            data = pickle.load(handle)

    mne_series: pd.Series = data
    return mne_series


def ita_series(
    indicator: str, frequency: Literal["A", "QNSA", "QSA"], from_cache=False, save_to_cache=True
) -> pd.Series:
    cache_fname = f"ita_{indicator}_{frequency}.pkl"
    if not from_cache:
        if "bea_key" not in api_keys:
            raise Exception("Couldn't find a bea_key entry in apiKeys.env")
        data = beaapi.get_data(
            userid=api_keys["bea_key"],
            datasetname="ITA",
            indicator=indicator,
            frequency=frequency,
            Year="ALL",
        )
        if save_to_cache:
            with open(cache_dir + "/" + cache_fname, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_dir + "/" + cache_fname, "rb") as handle:
            data = pickle.load(handle)

    ita_series: pd.Series = data.set_index("TimePeriod").DataValue
    ita_series.index = pd.to_datetime(ita_series.index).to_period(freq=frequency)
    return ita_series


def fred_series(
    series: str,
    start: str,
    frequency: Literal["A", "Q"],
    aggregation_method: str = None,
    from_cache=False,
    save_to_cache=True,
) -> pd.Series:
    cache_fname = "fred_" + series + "_" + start + ".pkl"
    if not from_cache:
        if "fred_key" not in api_keys:
            raise Exception("Couldn't find a fred_key entry in apiKeys.env")
        fred_series = Fred(api_keys["fred_key"]).get_series(
            series_id=series, observation_start=start
        )
        if save_to_cache:
            with open(cache_dir + "/" + cache_fname, "wb") as handle:
                pickle.dump(fred_series, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_dir + "/" + cache_fname, "rb") as handle:
            fred_series = pickle.load(handle)

    fred_series.index = fred_series.index.to_period(freq=frequency).rename("TimePeriod")
    if aggregation_method:
        fred_series = fred_series.groupby("TimePeriod").agg(aggregation_method)
    return fred_series


def excel_series(  # for local
    sheet: Union[str, int],
    index: str,
    column: str,
    first_row: int,  # 0-indexed!
    last_row: int,
    frequency: Literal["A", "Q"] = None,
    file: str = None,
    full_file=None,
) -> pd.Series:
    if full_file is None:
        file_paths = dotenv_values("localFiles.env")
        full_file = file_paths[file]
    excel_df = pd.read_excel(
        io=full_file,
        sheet_name=sheet,
        header=None,
        usecols=f"{index},{column}",
        skiprows=lambda x: x not in range(first_row, last_row + 1),
        dtype=str,
    )
    excel_df.columns = [0, 1]
    excel_df = excel_df.set_index(0, drop=True)
    if frequency is not None:
        if frequency == "A":
            excel_df.index = excel_df.index.str[:4]
        excel_df.index = (
            pd.to_datetime(excel_df.index).to_period(freq=frequency).rename("TimePeriod")
        )
    excel_series = excel_df[1]
    return excel_series


def cache_excel_file(url: str, from_cache=False, save_to_cache=True) -> str:
    if from_cache or save_to_cache:  # convert to local file
        cache_fname = os.path.basename(urlparse(url).path)
        full_file = cache_dir + "/" + cache_fname
        if not from_cache:
            data = requests.get(url)
            output = open(full_file, "wb")
            output.write(data.content)
            output.close()
        return full_file
    return url


def online_excel_series(
    url: str,
    sheet: Union[str, int],
    index: str,
    column: str,
    first_row: int,  # 0-indexed!
    last_row: int,
    frequency: Literal["A", "Q"] = None,
    from_cache=False,
    save_to_cache=True,
) -> pd.DataFrame:
    full_file = cache_excel_file(url, from_cache, save_to_cache)
    return excel_series(
        sheet=sheet,
        index=index,
        column=column,
        first_row=first_row,
        last_row=last_row,
        frequency=frequency,
        full_file=full_file,
    )


def imf_data(series: str, from_cache=False, save_to_cache=True) -> pd.DataFrame:
    cache_fname = "imf_" + series + ".pkl"
    if not from_cache:
        data = requests.get(f"https://www.imf.org/external/datamapper/api/v1/{series}").json()

        try:
            data = pd.DataFrame.from_dict(data.get("values")[series])
        except Exception as e:
            print("Error accessing data in IMF response")
            print(data)
            raise e

        if save_to_cache:
            with open(cache_dir + "/" + cache_fname, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_dir + "/" + cache_fname, "rb") as handle:
            data = pickle.load(handle)

    data.index = pd.to_datetime(data.index).to_period(freq="A").rename("TimePeriod")

    return data


def bls_data(
    series_dict: dict, start_year: "str", end_year: "str", from_cache=False, save_to_cache=True
) -> pd.DataFrame:
    # Need unique cache_fname from series_dict (and years)
    dhash = hashlib.md5()  # Use instead of hash() as that's not statble across runs
    dhash.update(json.dumps(series_dict, sort_keys=True).encode())
    h = dhash.hexdigest()
    cache_fname = f"bls_{h}_{start_year}_{end_year}.pkl"

    if not from_cache:
        if "bls_key" not in api_keys:
            raise Exception("Couldn't find a bls_key entry in apiKeys.env")
        url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

        # API key in config.py which contains: bls_key = 'key'
        key = "?registrationkey={}".format(api_keys["bls_key"])
        headers = {"Content-type": "application/json"}

        data = json.dumps(
            {"seriesid": list(series_dict.keys()), "startyear": start_year, "endyear": end_year}
        )

        # Post request for the data
        p_ret = requests.post("{}{}".format(url, key), headers=headers, data=data).json()
        try:
            p = p_ret["Results"]["series"]
        except Exception as e:
            print(f"Error accessing data in BLS response: {p_ret}")
            raise e
        if save_to_cache:
            with open(cache_dir + "/" + cache_fname, "wb") as handle:
                pickle.dump(p, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_dir + "/" + cache_fname, "rb") as handle:
            p = pickle.load(handle)

    # Date index from first series
    date_list = [f"{i['year']}" for i in p[0]["data"]]
    # Empty dataframe to fill with values
    bls_data = pd.DataFrame()
    # Build a pandas series from the API results, p
    for s in p:
        bls_data[series_dict[s["seriesID"]]] = pd.Series(
            index=pd.to_datetime(date_list, format="%Y"),
            data=[i["value"] for i in s["data"]],
            dtype="float64",
        )

    return bls_data
