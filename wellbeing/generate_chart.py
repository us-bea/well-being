# Note a few a checks (tagged HARDCODE) are hard-coded according to current API data availability.
# Can ignore flake8 F821 for undefined name 'config' as it's only a global access
from typing import List, Any, Union, Dict, Tuple
import yaml
import math
import datetime
import re
from importlib.resources import files

import pandas as pd
import numpy as np
import bokeh.plotting
import bokeh.models
import bokeh.io
import bokeh.themes
import bokeh.layouts
import bokeh.transform
from bokeh.models import BoxAnnotation
from . import get_data

dotenv_values = get_data.dotenv_values_lcl

precision_rules = {
    "index_values": 3,
    "growth_rate_pct": 1,
    "growth_rate_pct_contributions": 2,
    "levels_billions": 1,
    "shares": 1,  # or ratio
    # "pc_levels": 0,
}


def _pd_format_precision(type: str, percent: bool = True):
    return "{:." + str(precision_rules[type]) + ("%" if percent else "") + "}"


BLUE: str = "#004C97"
ORANGE: str = "#D86018"
GRAY: str = "#DCDEDF"  # light-gray
# Medium_gray= '#C1C4C5' # medium
DARK_GRAY: str = "#9EA2A2"  # gray

config: Dict[str, Any]


def _load_chart_config():
    global config
    with open(files(__package__).joinpath("chart_config.yml"), encoding="utf8") as f:
        config = yaml.safe_load(f)


_load_chart_config()


theme_path = files(__package__).joinpath("bokeh_theme.yml")


def _load_theme():
    bokeh.io.curdoc().theme = bokeh.themes.Theme(theme_path)


_load_theme()


def _period_float(period: pd.Period) -> float:
    """
    Turns a pandas Period into a float, properly scaled for bokeh. May become
    unnecessary with bokeh 3.1, as this is projected to include support for
    datetime units being used for things like the left and right parameters
    of BoxAnnotations
    """
    return period.to_timestamp().timestamp() * 1000


def _ts_float(ts) -> float:
    return ts.timestamp() * 1000


def _year_ticks(start_year: pd.Period, end_year: pd.Period, gap: int) -> list:
    return [
        _period_float(pd.Period(x))
        for x in range(start_year.year, end_year.year + 1)
        if (x - start_year.year) % gap == 0
    ]


def _make_range(series: pd.Series, round: int, pad: int) -> bokeh.models.Range1d:
    return bokeh.models.Range1d(
        math.floor(series.min() / round) * round - pad,
        math.ceil(series.max() / round) * round + pad,
    )


def _recession_shading(
    start_period: pd.Period, end_period: pd.Period, from_cache=False, save_to_cache=True
) -> List[BoxAnnotation]:
    frequency = start_period.freqstr.split("-")[0]
    recessions = get_data.recessions(
        frequency=frequency, from_cache=from_cache, save_to_cache=save_to_cache
    )
    recession_shading = [
        BoxAnnotation(
            left=_ts_float(max(start_period.to_timestamp(), recessions.loc[pi, "start"])),
            right=_ts_float(min(end_period.to_timestamp(how="end"), recessions.loc[pi, "stop"])),
            fill_color=GRAY,  #
            line_color=GRAY,
            # line_alpha=1,
            level="underlay",
            line_width=0,
        )
        for pi in recessions.index
        if recessions.loc[pi, "stop"] >= start_period.to_timestamp()
        and recessions.loc[pi, "start"] <= end_period.to_timestamp(how="end")
    ]
    return recession_shading


# Couldn't figure out how to override the stle of the existing tooltip, changing style
# Doing something like ".bk-tooltip-row-label {color: black !important;}" as a
# globalinlinestylesheet or as a stylesheet for the figure did not work. Plain style
# added (no selector) added to figure didn't work either
def _custom_tooltip(tt_orig):
    tt_new = (
        [
            '<div class="bk-tooltip-content">',
            "<div>",
            '<div style="display: table; border-spacing: 2px;">',
        ]
        + [
            '<div style="display: table-row;">'
            + '<div class="bk-tooltip-row-value" style="display: table-cell;">'
            + row_lbl
            + ': </div><div class="bk-tooltip-row-value" style="display: table-cell;">'
            + '<span data-value="">'
            + row_val
            + '</span><span class="bk-tooltip-color-block" data-swatch="" style="display: none;">'
            + " </span></div></div>"
            for (row_lbl, row_val) in tt_orig
        ]
        + ["</div>", "</div>", "</div>"]
    )
    return "\n".join(tt_new)


# Bokeh's figure titles can't easily handle subtitles, so add those separately as Div's.
def _format_notes(notes, **kwargs):
    notes = notes if isinstance(notes, list) else [notes]
    notes_text = (
        '<ul class="footnote">\n'
        + "\n".join(["<li>" + note.format(**kwargs) + "</li>" for note in notes])
        + "\n</ul>"
    )
    notes = bokeh.models.Div(
        text=notes_text,
        css_classes=["list-bordered", "fignotes"],
    )
    return notes


def get_narrative(section: str):
    """Get the narrative for a section

    Args:
        section (str): the section in the chart_config.yml
    """
    return (config["narrative date"], config[section]["narrative"])


def _add_metatext(
    chart: bokeh.plotting.figure, chart_config: dict, do_scale: bool = False, **kwargs
) -> bokeh.layouts.Column:
    column_elements = []

    if title_text := chart_config.get("title"):
        title_text_fmt = title_text.format(**kwargs)
        title = bokeh.models.Div(text=title_text_fmt, css_classes=["figtitle"])
        column_elements.append(title)
    if subtitle_text := chart_config.get("subtitle"):
        subtitle_text_fmt = subtitle_text.format(**kwargs)
        subtitle = bokeh.models.Div(text=subtitle_text_fmt, css_classes=["figtitle", "figsubtitle"])
        column_elements.append(subtitle)

    column_elements.append(chart)

    if notes := chart_config.get("notes"):
        notes = _format_notes(notes, **kwargs)
        column_elements.append(notes)

    if not do_scale:
        width = f'{chart_config["size"]["width"]}px'
    else:
        width = "98%"

    return bokeh.layouts.column(
        column_elements,
        styles={"width": width, "background-color": config["layout config"]["background_color"]},
        # stylesheets=[], # could use *ImportedStyleSheet to style these!
    )


def _write_xlsx_sheet(chart_df: pd.DataFrame, sheet_name: str):
    import os.path

    fname = dotenv_values("localFiles.env").get("output_xlsx", "")
    if fname != "":
        mode = "a" if os.path.isfile(fname) else "w"
        if_sheet_exists = "replace" if os.path.isfile(fname) else None
        with pd.ExcelWriter(
            fname, mode=mode, if_sheet_exists=if_sheet_exists, datetime_format="YYYY-MM-DD"
        ) as xlsx_writer:
            chart_df.to_excel(xlsx_writer, sheet_name=sheet_name)


def gdp(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    index_year: Union[int, str, pd.Period] = None,
    mid1=1973,
    mid2=2007,
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> Tuple[bokeh.layouts.Column, str]:
    """GDP and GDP Per Capita

    Creates a graph of GDP and GDP per capita relative to an index year and
    table showing growth from start_year to mid1, mid1 to mid2, and mid2 to end_year.

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None, which uses
          the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None, which uses the
          value in chart_config.yml.
        index_year (Union[int,str,pd.Period], optional): Index year used to make levels relative.
          Defaults to None, which uses the value in chart_config.yml.
        mid1 (int, optional): First mid point for table. Defaults to 1973.
        mid2 (int, optional): Second mid point for table. Defaults to 2007.
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        Tuple[bokeh.layouts.Column, str]: (Column layout of the graph and metadata,
          html rendering of growth table)
    """
    chart_config = config["Growth in Real GDP and Real GDP per Capita"]

    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    index_year = pd.Period(index_year or chart_config["defaults"]["index_year"])
    mid1 = pd.Period(mid1)
    mid2 = pd.Period(mid2)
    if mid1 <= start_year or mid2 <= mid1 or end_year <= mid2:
        raise Exception("Need start_year<mid1<mid2<end_year.")

    gdp = get_data.nipa_series(
        **chart_config["real GDP"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    gdp_pc = get_data.nipa_series(
        **chart_config["real GDP per capita"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    if min(start_year, index_year) < gdp.index.min():
        raise Exception(f"start_year and index_year needs to be at least {gdp.index.min()}")
    if max(end_year, index_year) > gdp.index.max():
        raise Exception(f"end_year and index_year needs to be at most {gdp.index.max()}")

    gdp = gdp[start_year:end_year]
    gdp_pc = gdp_pc[start_year:end_year]
    indexed_gdp = gdp / gdp[index_year]
    indexed_gdp_pc = gdp_pc / gdp_pc[index_year]

    chart_df = pd.concat(
        [indexed_gdp.rename("gdp"), indexed_gdp_pc.rename("gdp_pc")], axis="columns"
    )
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)
    _write_xlsx_sheet(
        pd.concat([gdp.rename("gdp_raw"), gdp_pc.rename("gdp_pc_raw"), chart_df], axis="columns"),
        "gdp",
    )

    chart = bokeh.plotting.figure(
        y_axis_label=f"Index numbers, [{index_year.year} = 1]",
        x_range=(start_year, end_year),
        y_range=(0, max(indexed_gdp.max(), indexed_gdp_pc.max())),
        x_axis_type="datetime",
        **chart_config["size"],
    )
    chart.xaxis.ticker = _year_ticks(start_year, end_year, gap=6)
    chart.xgrid.grid_line_color = None
    chart.ygrid.grid_line_color = GRAY
    for recession in _recession_shading(
        start_year, end_year, from_cache=from_cache, save_to_cache=save_to_cache
    ):
        chart.add_layout(recession)
    chart.xgrid.grid_line_color = None

    gdp_line = chart.line("TimePeriod", "gdp", name="gdp", color=BLUE, source=chart_data)
    gdp_pc_line = chart.line("TimePeriod", "gdp_pc", name="gdp_pc", color=ORANGE, source=chart_data)

    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("Real GDP", [gdp_line]),
                ("Real GDP per capita", [gdp_pc_line]),
                ("Recessions", [chart.vbar(color=GRAY)]),
            ]
        ),
        place="below",
    )

    tt = [("Year", "@TimePeriod{%Y}"), ("Index", "@$name")]
    tt = _custom_tooltip(tt)
    ht = bokeh.models.HoverTool(renderers=[gdp_line, gdp_pc_line], tooltips=tt)

    chart.add_tools(ht)

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    # Make table
    tbl_df = pd.concat([gdp.rename("gdp"), gdp_pc.rename("gdp_pc")], axis="columns")
    tbl_df = tbl_df.loc[[start_year, mid1, mid2, end_year]]
    tbl_df["Year"] = tbl_df.index.year
    tbl_df["real_gdp"] = (tbl_df["gdp"] / tbl_df["gdp"].shift(1)) ** (
        1 / (tbl_df["Year"] - tbl_df["Year"].shift(1))
    ) - 1
    tbl_df["real_gdppc"] = (tbl_df["gdp_pc"] / tbl_df["gdp_pc"].shift(1)) ** (
        1 / (tbl_df["Year"] - tbl_df["Year"].shift(1))
    ) - 1
    tbl_df = tbl_df.drop(pd.Period(start_year, freq="Y"), axis=0).drop(
        ["gdp", "gdp_pc", "Year"], axis=1
    )
    tbl_df.index = [f"{start_year}–{mid1}", f"{mid1}–{mid2}", f"{mid2}–{end_year}"]
    tbl_df.columns = ["Real GDP", "Real GDP per capita"]

    # Do explicit conversion to HTML to match previous constructions
    tbl_title = chart_config["table"]["title"]
    tbl_html = (
        tbl_df.reset_index()
        .style.format(
            {
                "Real GDP": _pd_format_precision("growth_rate_pct"),
                "Real GDP per capita": _pd_format_precision("growth_rate_pct"),
            }
        )
        .set_table_attributes('class="table table-condensed"')
        .relabel_index(["", "Real GDP", "Real GDP per capita"], axis=1)
        .hide(axis="index")
        .to_html()
    )
    tbl_html = tbl_html.replace(
        "<thead>",
        "<thead>"
        + f'<tr><th>&nbsp;</th><th class="text-center" colspan="2">{tbl_title}'
        + "<!--<br>[Percent]--></th></tr>",
    ).replace("%", '<span class="percent">%</span>')
    tbl_html = re.sub(
        '<td([^>]+)" >', '<td\\1 text-center" >', tbl_html
    )  # asscumes its <td id="XXX" class="XXX"

    return layout, tbl_html


def gdppc_comparison(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    countries: list = None,
    y_range=(0, 80000),
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """Per Capita Comparison with G-7 Developed Economies and Selected Other Countries

    Compares GDP per capital between start_year and end_year across countries.

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        countries (list, optional): None. Defaults to None, which uses the value in
          chart_config.yml.
        y_range (tuple, optional): Y-axis range for graphing. Defaults to (0, 80000).
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["GDP per Capita Country Comparisons"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    countries = countries or chart_config["defaults"]["countries"]
    countries_full = chart_config["defaults"]["countries_full"]

    if end_year <= start_year:
        raise Exception("Need start_year<end_year.")

    gdp_pc = get_data.nipa_series(
        table="T70100", frequency="A", line=1, from_cache=from_cache, save_to_cache=save_to_cache
    )
    imf_data = get_data.imf_data(
        series="NGDPDPC", from_cache=from_cache, save_to_cache=save_to_cache
    )
    if start_year < imf_data.index.min():
        raise Exception(f"start_year needs to be at least {imf_data.index.min()}")
    if end_year > imf_data.index.max():
        raise Exception(f"end_year needs to be at most {imf_data.index.max()}")

    gdp_pc = pd.DataFrame(gdp_pc[start_year:end_year])
    imf_data = imf_data[
        [
            "USA",
            "CAN",
            "DEU",
            "GBR",
            "FRA",
            "JPN",
            "ITA",
            "CHN",
            "RUS",
            "MEX",
            "TUR",
            "BRA",
            "IND",
            "IDN",
        ]
    ][start_year:end_year]

    imf_data["USA"] = gdp_pc["DataValue"].values

    imf_data = (
        imf_data.reset_index()
        .melt(id_vars=["TimePeriod"])
        .pivot(columns="TimePeriod", index="variable", values="value")[[start_year, end_year]]
        .reset_index()
        .rename(columns={"variable": "Country"})
        .sort_values(by=end_year, ascending=False)
        .rename(columns=lambda x: x if isinstance(x, str) else x.strftime("%Y"))
    )

    imf_data["country_remap"] = imf_data["Country"].map(
        chart_config["defaults"]["countries_rename"]
    )

    chart_data = bokeh.plotting.ColumnDataSource(imf_data)
    _write_xlsx_sheet(imf_data, "gdppc_comparison")

    chart = bokeh.plotting.figure(
        x_range=countries_full, y_range=y_range, y_axis_label="U.S. dollars", **chart_config["size"]
    )
    chart.ygrid.grid_line_color = GRAY

    start_year_bars = chart.vbar(
        x=bokeh.transform.dodge("country_remap", -0.25, range=chart.x_range),
        top=f"{start_year}",
        source=chart_data,
        width=0.25,
        color=BLUE,
    )
    end_year_bars = chart.vbar(
        x=bokeh.transform.dodge("country_remap", 0, range=chart.x_range),
        top=f"{end_year}",
        source=chart_data,
        width=0.25,
        color=ORANGE,
    )

    tt1 = [(f"GDP/Capita ({start_year})", f"@{start_year}{{$0,0}}")]
    tt1 = _custom_tooltip(tt1)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[start_year_bars],
            tooltips=tt1,
        )
    )

    tt2 = [(f"GDP/Capita ({end_year})", f"@{end_year}{{$0,0}}")]
    tt2 = _custom_tooltip(tt2)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[end_year_bars],
            tooltips=tt2,
        )
    )

    chart.add_layout(
        bokeh.models.Legend(
            items=[(f"{start_year}", [start_year_bars]), (f"{end_year}", [end_year_bars])]
        ),
        place="below",
    )

    chart.xaxis.major_label_orientation = 0.75
    chart.ygrid.grid_line_color = GRAY
    chart.xaxis.axis_line_color = None
    chart.yaxis.axis_line_color = None
    chart.yaxis.formatter = bokeh.models.NumeralTickFormatter(format="$0,0")
    chart.outline_line_color = None

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def income_growth_and_distribution(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    table_start: Union[int, str, pd.Period] = None,
    from_cache: bool = False,
    save_to_cache: bool = True,
    **kwargs,
) -> Tuple[bokeh.layouts.Column, str]:
    """Income Growth and its Distribution

    Graphs real GDP per capita and real median equivalized personal income between start_year
    and end_year. Creates a table of the overall change for these series from start_start to
    end_year.

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year for graph. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None, which uses the
          value in chart_config.yml.
        table_start (Union[int,str,pd.Period], optional): Start year for table. Defaults to None,
          which uses the value in chart_config.yml.
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        Tuple[bokeh.layouts.Column, str]: (Column layout of the graph and metadata, html of table)
    """
    broken = kwargs.get("broken", None)
    chart_config = config["Real Income Growth and Distribution"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    table_start = pd.Period(table_start or chart_config["defaults"]["table_start"])
    if table_start < start_year or end_year <= table_start:
        raise Exception("Need start_year<=table_start<end_year.")

    gdp_pc = get_data.nipa_series(
        **chart_config["Real GDP per capita"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    median_income = get_data.excel_series(
        **chart_config["Real median equivalized personal income"]
    ).astype("float")
    if start_year < median_income.index.min():
        raise Exception(f"start_year needs to be at least {median_income.index.min()}")
    if end_year > median_income.index.max():
        raise Exception(f"end_year needs to be at most {median_income.index.max()}")

    gdp_pc = gdp_pc[start_year:end_year]
    median_income = median_income[start_year:end_year]

    chart_df = pd.concat(
        [gdp_pc.rename("gdp_pc"), median_income.rename("median_income")], axis="columns"
    )
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)
    _write_xlsx_sheet(chart_df, "income_growth_and_distribution")

    chart = bokeh.plotting.figure(
        y_axis_label=chart_config["y_axis_label"],
        x_range=(start_year, end_year),
        x_axis_type="datetime",
        **chart_config["size"],
    )

    gdp_pc_line = chart.line("TimePeriod", "gdp_pc", name="gdp_pc", color=BLUE, source=chart_data)
    median_income_line = chart.line(
        "TimePeriod", "median_income", name="median_income", color=ORANGE, source=chart_data
    )

    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("Real GDP per capita", [gdp_pc_line]),
                ("Real median equivalized personal income", [median_income_line]),
                ("Recessions", [chart.vbar(color=GRAY)]),
            ]
        ),
        place="below",
    )
    for recession in _recession_shading(
        start_year, end_year, from_cache=from_cache, save_to_cache=save_to_cache
    ):
        chart.add_layout(recession)

    chart.xaxis.ticker = _year_ticks(start_year, end_year, gap=10)
    if broken is None:
        chart.yaxis.formatter = bokeh.models.NumeralTickFormatter(format="0,0")
    else:
        chart.y_range = bokeh.models.Range1d(
            min(broken), max(float(max(broken)), gdp_pc.max(), median_income.max())
        )
        chart.yaxis.ticker = broken
        chart.yaxis.major_label_overrides = {broken[0]: "0", broken[1]: "≈"}

    chart.ygrid.grid_line_color = GRAY
    chart.xaxis.axis_line_color = None
    chart.yaxis.axis_line_color = None
    chart.outline_line_color = None

    tt = [("Year", "@TimePeriod{%Y}"), ("Value", "@$name{$0,0}")]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[gdp_pc_line, median_income_line],
            tooltips=tt,
        )
    )

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    # Make table
    chart2 = pd.concat(
        [gdp_pc.rename("gdp_pc"), median_income.rename("median_income")], axis="columns"
    )
    chart2["median_income"] = chart2["median_income"].astype(float)
    chart2 = chart2.loc[[table_start, end_year]]
    chart2["Year"] = chart2.index.year
    chart2["gdp_pc"] = (chart2["gdp_pc"] / chart2["gdp_pc"].shift(1)) ** (
        1 / (chart2["Year"] - chart2["Year"].shift(1))
    ) - 1
    chart2["median_income"] = (chart2["median_income"] / chart2["median_income"].shift(1)) ** (
        1 / (chart2["Year"] - chart2["Year"].shift(1))
    ) - 1
    chart2 = chart2.drop(pd.Period(table_start, freq="Y"), axis=0).drop(["Year"], axis=1)
    chart2.index = [f"{table_start}–{end_year}"]
    chart2.columns = ["Real GDP per capita", "Real median equivalized personal income"]

    # Explicitly convert to HTML
    tbl_title = "Average Change"
    tbl_html = (
        chart2.reset_index()
        .style.format(
            {
                "Real median equivalized personal income": _pd_format_precision("growth_rate_pct"),
                "Real GDP per capita": _pd_format_precision("growth_rate_pct"),
            }
        )
        .set_table_attributes('class="table table-condensed"')
        .relabel_index(
            ["", "Real GDP<br>per capita", "Real median<br>equivalized<br>personal income"],
            axis=1,
        )
        .hide(axis="index")
        .to_html()
    )
    tbl_html = tbl_html.replace(
        "<thead>",
        "<thead>" + f'<tr><th>&nbsp;</th><th class="text-center" colspan="2">{tbl_title}</th></tr>',
    ).replace("%", '<span class="percent">%</span>')
    tbl_html = tbl_html = re.sub('<td([^>]+)col0" >', '<td\\1col0 text-center" >', tbl_html)
    tbl_html = tbl_html = re.sub(
        '<td([^>]+)col1" >', '<td\\1col1 text-center text-blue" >', tbl_html
    )
    tbl_html = tbl_html = re.sub(
        '<td([^>]+)col2" >', '<td\\1col2 text-center text-orange" >', tbl_html
    )

    return layout, tbl_html


def income_shares(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    y_range=(30, 70),
    from_cache: bool = False,
    save_to_cache: bool = True,
    **kwargs,
) -> bokeh.layouts.Column:
    """Distribution of Income Between Labor and Capital

    Graphs the labor and capital shares of income.

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        y_range (tuple, optional): Y-axis range. Defaults to (30, 70).
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    # We hide callouts in kwargs.
    chart_config = config["Income Shares"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    callouts = kwargs.get("callouts", chart_config["defaults"]["callouts"])
    callouts = list(map(pd.Period, callouts))
    broken = kwargs.get("broken", None)
    if end_year < start_year:
        raise Exception("Need start_year<end_year.")

    gdi = get_data.nipa_series(
        **chart_config["GDI"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    labor = get_data.nipa_series(
        **chart_config["Compensation of employees"],
        from_cache=from_cache,
        save_to_cache=save_to_cache,
    )
    proprietors = get_data.nipa_series(
        **chart_config["Proprietors income"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    if start_year < proprietors.index.min():
        raise Exception(f"start_year needs to be at least {proprietors.index.min()}")
    if end_year > proprietors.index.max():
        raise Exception(f"end_year needs to be at most {proprietors.index.max()}")
    gdi = gdi[start_year:end_year]
    labor = labor[start_year:end_year]
    proprietors = proprietors[start_year:end_year]
    gdi_minus_proprietors = gdi - proprietors
    capital = gdi_minus_proprietors - labor

    capital_share = (capital / gdi_minus_proprietors) * 100
    labor_share = (labor / gdi_minus_proprietors) * 100

    chart_df = pd.concat(
        [capital_share.rename("capital"), labor_share.rename("labor")], axis="columns"
    )
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)
    _write_xlsx_sheet(
        pd.concat(
            [gdi.rename("gdi"), labor.rename("labor"), proprietors.rename("proprietors"), chart_df],
            axis="columns",
        ),
        "income_shares",
    )

    def _callout_labels(series: pd.Series) -> pd.Series:
        return series.index.astype(str) + ",\n" + series.apply(lambda x: f"{x:.1f}%")

    lbl_x_offset = []
    if len(callouts) > 0:
        max_callout = max(callouts)
        lbl_x_offset = [0 if callout < pd.Period(max_callout - 1) else -7 for callout in callouts]
        callout_df = pd.concat(
            [capital_share[callouts].rename("capital_y"), labor_share[callouts].rename("labor_y")],
            axis="columns",
        ).assign(
            capital_label=_callout_labels(capital_share[callouts]),
            labor_label=_callout_labels(labor_share[callouts]),
            lbl_x_offset=lbl_x_offset,
        )
        callout_data = bokeh.plotting.ColumnDataSource(callout_df)

    chart = bokeh.plotting.figure(
        y_axis_label="Percent",
        y_range=y_range,
        x_range=(start_year, end_year + 1),
        x_axis_type="datetime",
        **chart_config["size"],
    )
    chart.ygrid.grid_line_color = GRAY

    capital_line = chart.line(
        "TimePeriod", "capital", name="capital", color=ORANGE, source=chart_data
    )
    labor_line = chart.line("TimePeriod", "labor", name="labor", color=BLUE, source=chart_data)
    if len(callouts) > 0:
        chart.scatter("TimePeriod", "capital_y", color=ORANGE, source=callout_data)
        chart.scatter("TimePeriod", "labor_y", color=BLUE, source=callout_data)

    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("Labor share of income", [labor_line]),
                ("Capital share of income", [capital_line]),
                ("Recessions", [chart.vbar(color=GRAY)]),
            ]
        ),
        place="below",
    )
    if len(callouts) > 0:
        chart.add_layout(
            bokeh.models.LabelSet(
                x="TimePeriod",
                y="capital_y",
                text="capital_label",
                x_offset="lbl_x_offset",
                y_offset=15,
                text_color=ORANGE,
                source=callout_data,
            )
        )
        chart.add_layout(
            bokeh.models.LabelSet(
                x="TimePeriod",
                y="labor_y",
                text="labor_label",
                x_offset="lbl_x_offset",
                y_offset=15,
                text_color=BLUE,
                source=callout_data,
            )
        )
    for recession in _recession_shading(
        start_year, end_year, from_cache=from_cache, save_to_cache=save_to_cache
    ):
        chart.add_layout(recession)

    tt = [
        ("Year", "@TimePeriod{%Y}"),
        ("Percent", "@$name{0." + "0" * precision_rules["shares"] + "}"),
    ]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[capital_line, labor_line],
            tooltips=tt,
        )
    )
    # ^^^ Would prefer a vline hover tool that gives both datapoints in one tooltip,
    # but it's nontrivial to set up.
    chart.xaxis.ticker = _year_ticks(start_year, end_year, 5)
    chart.xaxis.axis_line_color = None
    chart.yaxis.axis_line_color = None
    chart.outline_line_color = None

    if broken is not None:
        chart.y_range = bokeh.models.Range1d(min(broken), max(broken))
        chart.yaxis.ticker = broken
        chart.yaxis.major_label_overrides = {broken[0]: "0", broken[1]: "≈"}

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def net_worth(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    from_cache: bool = False,
    save_to_cache: bool = True,
    **kwargs,
) -> bokeh.layouts.Column:
    """Trends in Household Wealth as Measured by Net Worth

    Graph the ratio of household net worth to DPI alongside household net worth.

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["Net Worth"]
    broken = kwargs.get("broken", None)
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    if end_year < start_year:
        raise Exception("Need start_year<end_year.")

    dpi = get_data.nipa_series(
        **chart_config["DPI"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    net_worth = get_data.fred_series(
        **chart_config["Net Worth"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    if start_year < net_worth.index.min():
        raise Exception(f"start_year needs to be at least {net_worth.index.min()}")
    if end_year > net_worth.index.max():
        raise Exception(f"end_year needs to be at most {net_worth.index.max()}")
    dpi = dpi[start_year:end_year]
    net_worth = net_worth[start_year:end_year]
    ratio = net_worth / dpi
    net_worth = net_worth / 1000  # convert from millions of $ to billions of $

    chart_df = pd.concat([net_worth.rename("net_worth"), ratio.rename("ratio")], axis="columns")
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)
    _write_xlsx_sheet(pd.concat([dpi.rename("dpi"), chart_df], axis="columns"), "net_worth")

    chart = bokeh.plotting.figure(
        y_range=_make_range(ratio, **chart_config["ranges"]["ratio"]),
        x_range=(start_year, end_year),
        x_axis_type="datetime",
        **chart_config["size"],
    )
    chart.xaxis.ticker = _year_ticks(start_year, end_year, 5)
    chart.ygrid.grid_line_color = GRAY

    chart.extra_y_ranges["net_worth_range"] = _make_range(
        net_worth, **chart_config["ranges"]["net_worth"]
    )
    net_worth_line = chart.line(
        "TimePeriod",
        "net_worth",
        name="net_worth",
        color=ORANGE,
        y_range_name="net_worth_range",
        source=chart_data,
    )
    ratio_line = chart.line("TimePeriod", "ratio", name="ratio", color=BLUE, source=chart_data)

    chart.yaxis.formatter = bokeh.models.NumeralTickFormatter(format="0%")
    if broken is not None:
        chart.y_range = bokeh.models.Range1d(min(broken), max(broken))
        chart.yaxis.ticker = broken
        chart.yaxis.major_label_overrides = {broken[0]: "0", broken[1]: "≈"}
    chart.yaxis.major_label_text_color = BLUE
    right_axis = bokeh.models.LinearAxis(
        y_range_name="net_worth_range",
        axis_label="Billions of dollars",  # Rescaled from millions of $ above
        formatter=bokeh.models.NumeralTickFormatter(format="0,0"),
        axis_label_text_color=ORANGE,
        major_label_text_color=ORANGE,
    )  # major_label_orientation https://github.com/bokeh/bokeh/pull/13044
    chart.add_layout(
        right_axis,
        place="right",
    )
    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("Ratio: Household net worth to DPI", [ratio_line]),
                ("Household net worth", [net_worth_line]),
                ("Recessions", [chart.vbar(color=GRAY)]),
            ]
        ),
        place="below",
    )

    for recession in _recession_shading(
        start_year, end_year, from_cache=from_cache, save_to_cache=save_to_cache
    ):
        chart.add_layout(recession)

    chart.xaxis.axis_line_color = None
    chart.yaxis.axis_line_color = None
    chart.outline_line_color = None

    tt1 = [
        ("Year", "@TimePeriod{%Y}"),
        ("Value", "@$name{0." + "0" * precision_rules["growth_rate_pct"] + "%}"),
    ]
    tt1 = _custom_tooltip(tt1)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[ratio_line],
            tooltips=tt1,
        )
    )
    tt2 = [
        ("Year", "@TimePeriod{%Y}"),
        ("Value", "@$name{$0,0." + "0" * precision_rules["levels_billions"] + "}"),
    ]
    tt2 = _custom_tooltip(tt2)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[net_worth_line],
            tooltips=tt2,
        )
    )

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def inflation_trends(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """Inflation Trends: Percent Changes in Consumer Prices

    Graphs percent change in PCE.

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["Inflation Trends"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    if end_year <= start_year:
        raise Exception("Need start_year<end_year.")

    prices_pc = get_data.nipa_series(
        **chart_config["Percent Change in PCE Prices"],
        from_cache=from_cache,
        save_to_cache=save_to_cache,
    )
    if start_year < prices_pc.index.min():
        raise Exception(f"start_year needs to be at least {prices_pc.index.min()}")
    if end_year > prices_pc.index.max():
        raise Exception(f"end_year needs to be at most {prices_pc.index.max()}")
    prices_pc = prices_pc[start_year:end_year]

    chart_df = pd.concat(
        [
            prices_pc.rename("prices_pc"),
        ],
        axis="columns",
    )
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)
    _write_xlsx_sheet(chart_df, "inflation_trends")

    chart = bokeh.plotting.figure(
        y_axis_label="Percent",
        x_range=(start_year - 1, end_year + 1),
        x_axis_type="datetime",
        **chart_config["size"],
    )

    bar_width = 250
    dodge_amount = bar_width / 2 * 24 * 60 * 60 * 1000
    pc_bars = chart.vbar(
        x=bokeh.transform.dodge("TimePeriod", dodge_amount),
        top="prices_pc",
        width=datetime.timedelta(days=bar_width),
        source=chart_data,
        line_width=0.0,
    )

    tt = [
        ("Year", "@TimePeriod{%Y}"),
        (
            "Percent Change in PCE Prices",
            "@prices_pc{0." + "0" * precision_rules["growth_rate_pct"] + "}",
        ),
    ]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[pc_bars],
            tooltips=tt,
        )
    )

    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("Percent change in PCE prices", [pc_bars]),
                ("Recessions", [chart.vbar(color=GRAY)]),
            ]
        ),
        place="below",
    )
    for recession in _recession_shading(
        start_year, end_year, from_cache=from_cache, save_to_cache=save_to_cache
    ):
        chart.add_layout(recession)

    chart.xaxis.ticker = _year_ticks(start_year, end_year, gap=10)
    chart.ygrid.grid_line_color = GRAY
    chart.xaxis.axis_line_color = None
    chart.yaxis.axis_line_color = None
    chart.outline_line_color = None

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def employment(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    y_range: Tuple = (20, 100),
    y_range2: Tuple = (2, 10),
    from_cache: bool = False,
    save_to_cache: bool = True,
    **kwargs,
) -> bokeh.layouts.Column:
    """Employment Trend

    Graph FTE employment rate, Prime age employment rate, and unemployment rate

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        y_range (Tuple, optional): Primary (left) y-axis range (percent). Defaults to (20, 100).
        y_range2 (Tuple, optional): Secondary (right) y-axis range (percent). Defaults to (2, 10).
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    # hide callouts in kwargs
    chart_config = config["Employment"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    broken1 = kwargs.get("broken1", None)
    broken2 = kwargs.get("broken2", None)
    callouts = list(map(pd.Period, kwargs.get("callouts", chart_config["defaults"]["callouts"])))
    if end_year <= start_year:
        raise Exception("Need start_year<end_year.")

    full_time_equivalent = (
        pd.concat(
            [
                get_data.nipa_series(
                    **chart_config[fte_table], from_cache=from_cache, save_to_cache=save_to_cache
                )
                for fte_table in [k for k in chart_config if k.startswith("Full-time Equivalent")]
            ]
        ).drop_duplicates()
    )[start_year:end_year]
    population = get_data.nipa_series(
        **chart_config["Population"], from_cache=from_cache, save_to_cache=save_to_cache
    )[start_year:end_year]
    prime_age = get_data.fred_series(
        **chart_config["Prime Age Employment"], from_cache=from_cache, save_to_cache=save_to_cache
    )[start_year:end_year]
    unemployment = get_data.fred_series(
        **chart_config["Unemployment"], from_cache=from_cache, save_to_cache=save_to_cache
    )[start_year:end_year]
    if start_year < population.index.min():
        raise Exception(f"start_year needs to be at least {population.index.min()}")
    if end_year > population.index.max():
        raise Exception(f"end_year needs to be at most {population.index.max()}")

    fte_employment = full_time_equivalent / population * 100  # Convert to numeral percent

    chart_df = pd.concat(
        [
            fte_employment.rename("fte_employment"),
            prime_age.rename("prime_age"),
            unemployment.rename("unemployment"),
        ],
        axis="columns",
    )
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)
    _write_xlsx_sheet(
        pd.concat([full_time_equivalent, population, chart_df], axis="columns"), "employment"
    )

    def _callout_labels(series: pd.Series) -> pd.Series:
        return series.index.astype(str) + ",\n" + series.apply(lambda x: f"{(100*x):.1f}%")

    if len(callouts) > 0:
        callout_data = bokeh.plotting.ColumnDataSource(
            pd.concat(
                [
                    fte_employment[callouts].rename("fte_employment_y"),
                    prime_age[callouts].rename("prime_age_y"),
                    unemployment[callouts].rename("unemployment_y"),
                ],
                axis="columns",
            ).assign(
                fte_employment_label=_callout_labels(fte_employment[callouts]),
                prime_age_label=_callout_labels(prime_age[callouts]),
                unemployment_label=_callout_labels(unemployment[callouts]),
            )
        )

    if broken1 is not None:
        y_range = (min(broken1), max(broken1))
    chart = bokeh.plotting.figure(
        y_axis_label="Employment rates, percent",
        y_range=y_range,
        x_range=(start_year, end_year),
        x_axis_type="datetime",
        **chart_config["size"],
    )
    if broken1 is not None:
        chart.yaxis.ticker = broken1
        chart.yaxis.major_label_overrides = {broken1[0]: "0", broken1[1]: "≈"}
    chart.ygrid.grid_line_color = GRAY

    fte_line = chart.line(
        "TimePeriod", "fte_employment", name="fte_employment", color=BLUE, source=chart_data
    )
    prime_age_line = chart.line(
        "TimePeriod",
        "prime_age",
        name="prime_age",
        color=BLUE,
        line_dash="dashed",
        source=chart_data,
    )
    if len(callouts) > 0:
        chart.scatter("TimePeriod", "fte_employment_y", color=BLUE, source=callout_data)
        chart.scatter("TimePeriod", "prime_age_y", color=BLUE, source=callout_data)

    if broken2 is not None:
        y_range2 = (min(broken2), max(broken2))
    chart.extra_y_ranges["unemployment"] = bokeh.models.Range1d(*y_range2)
    unemployment_line = chart.line(
        "TimePeriod",
        "unemployment",
        name="unemployment",
        y_range_name="unemployment",
        color=ORANGE,
        source=chart_data,
    )
    if len(callouts) > 0:
        chart.scatter(
            "TimePeriod",
            "unemployment_y",
            y_range_name="unemployment",
            color=ORANGE,
            source=callout_data,
        )

    chart.yaxis.axis_label_text_color = BLUE
    chart.yaxis.major_label_text_color = BLUE
    if len(callouts) > 0:
        chart.add_layout(
            bokeh.models.LabelSet(
                x="TimePeriod",
                y="fte_employment_y",
                text="fte_employment_label",
                x_offset=-15,
                y_offset=15,
                text_color=BLUE,
                source=callout_data,
            )
        )
        chart.add_layout(
            bokeh.models.LabelSet(
                x="TimePeriod",
                y="prime_age_y",
                text="prime_age_label",
                x_offset=-15,
                y_offset=15,
                text_color=BLUE,
                source=callout_data,
            )
        )
    right_axis = bokeh.models.LinearAxis(
        y_range_name="unemployment",
        axis_label="Unemployment rate, percent",
        axis_label_text_color=ORANGE,
        major_label_text_color=ORANGE,
    )
    if broken2 is not None:
        # right_axis.range = bokeh.models.Range1d(min(broken2), max(broken2))
        right_axis.ticker = broken2
        right_axis.major_label_overrides = {broken2[0]: "0", broken2[1]: "≈"}
    chart.add_layout(
        right_axis,
        place="right",
    )
    if len(callouts) > 0:
        chart.add_layout(
            bokeh.models.LabelSet(
                x="TimePeriod",
                y="unemployment_y",
                text="unemployment_label",
                x_offset=-20,
                y_offset=-20,
                y_range_name="unemployment",
                text_color=ORANGE,
                source=callout_data,
            )
        )
    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("FTE employment rate (BEA)", [fte_line]),
                ("Prime age employment rate (BLS)", [prime_age_line]),
                ("Unemployment rate (BLS)", [unemployment_line]),
                ("Recessions", [chart.vbar(color=GRAY)]),
            ]
        ),
        place="below",
    )
    for recession in _recession_shading(
        start_year, end_year, from_cache=from_cache, save_to_cache=save_to_cache
    ):
        chart.add_layout(recession)

    tt = [
        ("Year", "@TimePeriod{%Y}"),
        ("Percent", "@$name{0." + "0" * precision_rules["shares"] + "}"),
    ]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[fte_line, prime_age_line, unemployment_line],
            tooltips=tt,
        )
    )

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def industry_growth(
    start_quarter: Union[str, pd.Period] = None,
    end_quarter: Union[str, pd.Period] = None,
    x_range: Tuple[float, float] = (-1, 1.45),
    x_ticker_range: Tuple[float, float] = (-0.4, 1.2),
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """Industry Comparison of Economic Growth

    Growth rate of industries between start_year and end_year.
    Adjust x_range to ensure industry labels are visible and adjusted horizontally.

    Args:
        start_quarter (Union[str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_quarter (Union[str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        x_range (Tuple[float,float], optional): X-axis rage. Defaults to (-1,1.45).
        x_ticker_range (Tuple[float,float], optional): Range over which to mark x-ticks.
          Defaults to (-0.4, 1.2).
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["Industry Comparisons"]
    start_quarter = pd.Period(start_quarter or chart_config["defaults"]["start_quarter"])
    end_quarter = pd.Period(end_quarter or chart_config["defaults"]["end_quarter"])
    if end_quarter <= start_quarter:
        raise Exception("Need start_quarter<end_quarter.")
    x_ticker_start, x_ticker_stop = x_ticker_range

    ind_data = (
        get_data.industry_data(
            **chart_config["Industry Data"], from_cache=from_cache, save_to_cache=save_to_cache
        )
        .query(f'Industry in {chart_config["industries"]}')
        .drop(columns="Industry")
    )
    if start_quarter < ind_data.index.min():
        raise Exception(f"start_quarter needs to be at least {ind_data.index.min()}")
    if end_quarter > ind_data.index.max():
        raise Exception(f"end_quarter needs to be at most {ind_data.index.max()}")
    ind_data = (
        ind_data.loc[[start_quarter, end_quarter]]
        .reset_index()
        .pivot(index="IndustryDescription", columns="TimePeriod")
        .droplevel(level=0, axis="columns")
        .reset_index()
        .rename(columns={"IndustryDescription": "industry"})
        .replace("Gross domestic product", "GDP")
        .assign(
            percent_change=lambda x: (x[end_quarter] / x[start_quarter])
            ** (4 / (end_quarter - start_quarter).n)
            - 1,
            label_alignment=lambda x: np.where(x.percent_change >= 0, "left", "right"),
            label_offset=lambda x: np.where(x.percent_change >= 0, 5, -5),
            bar_color=lambda x: np.where(x.industry == "GDP", ORANGE, BLUE),
        )
        .sort_values("percent_change")
        .drop(columns=[start_quarter, end_quarter])
    )

    chart_data = bokeh.plotting.ColumnDataSource(ind_data)
    _write_xlsx_sheet(ind_data, "industry_growth")

    chart = bokeh.plotting.figure(
        y_axis_label="Industries",
        y_range=ind_data["industry"].tolist(),
        x_range=x_range,
        **chart_config["size"],
    )
    chart.yaxis.visible = False
    chart.xaxis.major_tick_line_color = None
    chart.xaxis.minor_tick_line_color = None

    ind_bars = chart.hbar(
        y="industry",
        right="percent_change",
        line_width=1,
        source=chart_data,
        line_color="white",
        color="bar_color",
    )

    tt = [
        ("Industry", "@industry"),
        ("Percent Change", "@percent_change{0." + "0" * precision_rules["growth_rate_pct"] + "%}"),
    ]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[ind_bars],
            tooltips=tt,
        )
    )

    # Generate labels that align on the right/left of bar, based on whether value is positive
    #  or negative
    labels = bokeh.models.LabelSet(
        x="percent_change",
        y="industry",
        text="industry",
        level="glyph",
        source=chart_data,
        text_font_size="10pt",
        y_offset=-8,
        text_align="label_alignment",
        x_offset="label_offset",
    )
    chart.add_layout(labels)

    gdp_change = ind_data[ind_data.industry == "GDP"]["percent_change"].values[0]

    chart.segment(
        x0=-100,
        y0=ind_data.loc[ind_data["industry"] == "GDP", "industry"],
        x1=gdp_change - 0.12 if gdp_change < 0 else 0,
        y1=ind_data.loc[ind_data["industry"] == "GDP", "industry"],
        line_dash="dashed",
        line_color=ORANGE,
        line_width=2,
    )

    chart.segment(
        x0=0 if gdp_change < 0 else gdp_change + 0.12,
        y0=ind_data.loc[ind_data["industry"] == "GDP", "industry"],
        x1=100,
        y1=ind_data.loc[ind_data["industry"] == "GDP", "industry"],
        line_dash="dashed",
        line_color=ORANGE,
        line_width=2,
    )

    chart.xaxis.ticker = np.arange(x_ticker_start, x_ticker_stop, 0.2)
    chart.xaxis[0].formatter = bokeh.models.NumeralTickFormatter(format="0%")

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_quarter=start_quarter, end_quarter=end_quarter
    )

    return layout


def state_income_growth(
    start_year: Union[str, pd.Period] = None,
    end_year: Union[str, pd.Period] = None,
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """State Comparison of Personal Income Growth

    Comparison of states' income growth from start_year to end_year.

    Args:
        start_year (Union[str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["State Real Income Growth"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    if end_year <= start_year:
        raise Exception("Need start_year<end_year.")

    state_inc = get_data.regional_data(
        "SARPI", "A", "STATE", 2, from_cache=from_cache, save_to_cache=save_to_cache
    )
    if start_year < state_inc.index.min():
        raise Exception(f"start_year needs to be at least {state_inc.index.min()}")
    if end_year > state_inc.index.max():
        raise Exception(f"end_year needs to be at most {state_inc.index.max()}")
    start_year = start_year.strftime("%Y")
    end_year = end_year.strftime("%Y")
    state_inc = (
        state_inc.loc[[start_year, end_year]]
        .reset_index()[["GeoName", "TimePeriod", "DataValue"]]
        .pivot(index="GeoName", columns="TimePeriod")
        .drop("District of Columbia", axis=0)
        .droplevel(level=0, axis="columns")
        .rename(columns=lambda x: x.strftime("%Y"))
        .reset_index()
    )

    state_inc["per_change"] = (state_inc[end_year] / state_inc[start_year]) ** (
        1 / (int(end_year) - int(start_year))
    ) - 1
    state_inc = state_inc.sort_values(by=["per_change"])

    state_inc["color_list"] = state_inc.apply(
        lambda x: "#D86018" if x["GeoName"] == "United States" else "#004c97", axis=1
    )

    chart_data = bokeh.plotting.ColumnDataSource(state_inc)
    _write_xlsx_sheet(state_inc, "state_income_growth")

    chart = bokeh.plotting.figure(
        y_axis_label=f"Percent change ({start_year}–{end_year})",
        x_range=state_inc["GeoName"].tolist(),
        **chart_config["size"],
    )
    chart.xaxis.major_label_orientation = math.pi / 2
    chart.yaxis.formatter = bokeh.models.NumeralTickFormatter(format="0.0%")

    pc_bars = chart.vbar(
        x="GeoName",
        top="per_change",
        width=0.9,
        source=chart_data,
        line_color="white",
        color="color_list",
    )

    tt = [
        ("State", "@GeoName"),
        ("Percent Change", "@per_change{0." + "0" * precision_rules["growth_rate_pct"] + "%}"),
    ]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[pc_bars],
            tooltips=tt,
        )
    )

    chart.ygrid.grid_line_color = GRAY
    chart.xaxis.axis_line_color = None
    chart.yaxis.axis_line_color = None
    chart.outline_line_color = None

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def state_income(
    year: Union[str, pd.Period] = None,
    y_range=(0, 80_000),
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """State Comparison of Income per capita

    Comparison of income per capita in year across states.

    Args:
        year (Union[str,pd.Period], optional): Year. Defaults to None,
          which uses the value in chart_config.yml.
        y_range (tuple, optional): Y-axis range. Defaults to (0, 80_000).
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["State Income Levels"]
    year = pd.Period(year or chart_config["defaults"]["year"])

    state_income = get_data.regional_data(
        **chart_config["State Income"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    if year < state_income.index.min():
        raise Exception(f"year needs to be at least {state_income.index.min()}")
    if year > state_income.index.max():
        raise Exception(f"year needs to be at most {state_income.index.max()}")
    state_income = (
        state_income.loc[year]
        .reset_index()[["GeoName", "TimePeriod", "DataValue"]]
        .pivot(index="GeoName", columns="TimePeriod")
        .drop("District of Columbia", axis=0)
        .droplevel(level=0, axis="columns")
        .sort_values(by=year)
        .rename(columns=lambda x: x.strftime("%Y"))
        .reset_index()
        .assign(bar_color=lambda x: np.where(x.GeoName == "United States", ORANGE, BLUE))
    )

    chart_data = bokeh.plotting.ColumnDataSource(state_income)
    _write_xlsx_sheet(state_income, "state_income")

    chart = bokeh.plotting.figure(
        y_axis_label=chart_config["y_axis_label"],
        x_range=state_income["GeoName"].tolist(),
        y_range=y_range,
        **chart_config["size"],
    )
    chart.xaxis.major_label_orientation = math.pi / 2

    pc_bars = chart.vbar(
        x="GeoName",
        top=year.strftime("%Y"),
        source=chart_data,
        width=0.9,
        line_color="white",
        color="bar_color",
    )

    tt = [("State", "@GeoName"), ("Real Income per Capita", f"@{year}{{$0,0}}")]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[pc_bars],
            tooltips=tt,
        )
    )

    chart.yaxis.formatter = bokeh.models.NumeralTickFormatter(format="0,0")
    chart.ygrid.grid_line_color = GRAY
    chart.xaxis.axis_line_color = None
    chart.yaxis.axis_line_color = None
    chart.outline_line_color = None

    layout = _add_metatext(chart=chart, chart_config=chart_config, year=year)

    return layout


def sustainable_growth(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """Sustainable Economic Growth: Real GDP vs Real Net Domestic Product

    Real GDP vs Real Net Domestic Product from start_year to end_year.

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["GDP vs NDP"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    if end_year <= start_year:
        raise Exception("Need start_year<end_year.")

    gdp = (
        get_data.nipa_series(
            **chart_config["GDP"], from_cache=from_cache, save_to_cache=save_to_cache
        )
        / 1000
    )  # Scale to billions
    ndp = (
        get_data.nipa_series(
            **chart_config["NDP"], from_cache=from_cache, save_to_cache=save_to_cache
        )
        / 1000
    )  # Scale to billions
    if start_year < ndp.index.min():
        raise Exception(f"start_year needs to be at least {ndp.index.min()}")
    if end_year > ndp.index.max():
        raise Exception(f"end_year needs to be at most {ndp.index.max()}")
    ndp = ndp[start_year:end_year]
    gdp = gdp[start_year:end_year]

    chart_df = pd.concat([gdp.rename("gdp"), ndp.rename("ndp")], axis="columns")
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)
    _write_xlsx_sheet(chart_df, "sustainable_growth")

    chart = bokeh.plotting.figure(
        y_axis_label=chart_config["y_axis_label"],
        x_range=(start_year, end_year),
        x_axis_type="datetime",
        **chart_config["size"],
    )

    gdp_line = chart.line("TimePeriod", "gdp", name="gdp", color=BLUE, source=chart_data)
    ndp_line = chart.line("TimePeriod", "ndp", name="ndp", color=ORANGE, source=chart_data)

    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("Real gross domestic product", [gdp_line]),
                ("Real net domestic product", [ndp_line]),
                ("Recessions", [chart.vbar(color=GRAY)]),
            ]
        ),
        place="below",
    )
    for recession in _recession_shading(
        start_year, end_year, from_cache=from_cache, save_to_cache=save_to_cache
    ):
        chart.add_layout(recession)

    tt = [
        ("Year", "@TimePeriod{%Y}"),
        ("Value", "@$name{$0,0." + "0" * precision_rules["levels_billions"] + "}"),
    ]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[gdp_line, ndp_line],
            tooltips=tt,
        )
    )
    # ^^^ Would prefer a vline hover tool that gives both datapoints in one tooltip,
    # but it's nontrivial to set up.
    chart.yaxis.formatter = bokeh.models.NumeralTickFormatter(format="0,0")
    chart.xaxis.ticker = _year_ticks(start_year, end_year, gap=10)
    chart.ygrid.grid_line_color = GRAY
    chart.xaxis.axis_line_color = None
    chart.yaxis.axis_line_color = None
    chart.outline_line_color = None

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def economic_growth_tables(
    start_year: Union[int, str, pd.Period] = None,
    mid_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> str:
    """Trends in Economic Growth

    Creates a table of economic growth from start_year to mid_year and mid_year to end_year.

    Args:
        start_year (Union[int,str,pd.Period]): Start year
        mid_year (Union[int,str,pd.Period]): Mid year
        end_year (Union[int,str,pd.Period]): End year
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). This function uses BLS queries and the cache only works
          if start_year and end_year remain the same. Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        str: HTML representation of Table
    """
    chart_config = config["Real Economic Growth Tables"]

    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    mid_year = pd.Period(mid_year or chart_config["defaults"]["mid_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    if mid_year <= start_year or end_year <= mid_year:
        raise Exception("Need start_year<mid_year<end_year.")
    if start_year < pd.Period(1987):  # HARDCODE
        raise Exception("Need start_year>=1987.")

    # Series stored as a dictionary
    series_dict = {
        "MPU4900012": "Total Factor Productivity",
        "MPU4900042": "Capital Input - Index",
        "MPU4900052": "Labor Input - Index",
        "MPU4900131": "Capital Share",
        "MPU4900141": "Labor Share",
    }

    # Get these in chunks because that's what the API caps
    spread = (end_year - start_year).n + 1
    bls_chunk = 10  #
    n_segs = int(spread / bls_chunk) + (spread % bls_chunk > 0)
    starts = [start_year + g * bls_chunk for g in range(n_segs)]
    ends = [start_year + (g + 1) * bls_chunk - 1 for g in range(n_segs - 1)] + [end_year]
    bls_pulls = [
        get_data.bls_data(
            series_dict=series_dict,
            start_year=str(start_seg),
            end_year=str(end_seg),
            from_cache=from_cache,
            save_to_cache=save_to_cache,
        )
        for (start_seg, end_seg) in zip(starts, ends)
    ]

    bls = pd.concat(bls_pulls)

    gdp = get_data.nipa_series(
        table="T10103", frequency="A", line=1, from_cache=from_cache, save_to_cache=save_to_cache
    )[start_year.strftime("%Y") : end_year.strftime("%Y")]
    gdp.index = gdp.index.to_timestamp()
    bls = (
        bls.merge(gdp, how="left", left_index=True, right_index=True)
        .rename(columns={"DataValue": "GDP"})
        .sort_index()
    )
    if end_year.to_timestamp() > bls.index.max():
        raise Exception(f"end_year needs to be at most {bls.index.max()}")
    start_year = start_year.strftime("%Y")
    mid_year = mid_year.strftime("%Y")
    end_year = end_year.strftime("%Y")
    _write_xlsx_sheet(bls, "economic_growth_tables")

    # For shares we think of the periods as (start, end] as the growth rate is about deltas
    # (fence-post problem)
    shares = (
        bls[["Capital Share", "Labor Share"]]
        .astype(float)
        .drop(start_year, axis=0)
        .assign(
            ColNames=lambda x: [
                f"{mid_year}-{end_year}" if idx.year > int(mid_year) else f"{start_year}-{mid_year}"
                for idx in x.index
            ]
        )
    )

    shares = (
        shares[["Capital Share", "Labor Share", "ColNames"]]
        .groupby("ColNames")
        .mean(numeric_only=True)
    )

    average_annual_growth = bls.loc[
        (bls.index == start_year) | (bls.index == mid_year) | (bls.index == end_year)
    ][["Total Factor Productivity", "Capital Input - Index", "Labor Input - Index", "GDP"]]
    average_annual_growth["Year"] = (
        pd.DatetimeIndex(average_annual_growth.index).strftime("%Y").astype(int)
    )

    average_annual_growth[
        ["Total Factor Productivity", "Capital Input - Index", "Labor Input - Index"]
    ] = average_annual_growth[
        ["Total Factor Productivity", "Capital Input - Index", "Labor Input - Index"]
    ].astype(
        float
    )

    average_annual_growth["tfp_lag"] = average_annual_growth["Total Factor Productivity"].shift(1)
    average_annual_growth["gdp_lag"] = average_annual_growth["GDP"].shift(1)
    average_annual_growth["cap_lag"] = average_annual_growth["Capital Input - Index"].shift(1)
    average_annual_growth["lab_lag"] = average_annual_growth["Labor Input - Index"].shift(1)
    average_annual_growth["year_lag"] = average_annual_growth.Year.shift(1)

    average_annual_growth = average_annual_growth.dropna()

    average_annual_growth["Real GDP"] = (
        (
            (average_annual_growth["GDP"] / average_annual_growth["gdp_lag"])
            ** (1 / (average_annual_growth["Year"] - average_annual_growth["year_lag"]))
        )
        - 1
    ) * 100
    average_annual_growth["Labor Input"] = (
        (
            (average_annual_growth["Labor Input - Index"] / average_annual_growth["lab_lag"])
            ** (1 / (average_annual_growth["Year"] - average_annual_growth["year_lag"]))
        )
        - 1
    ) * 100
    average_annual_growth["Capital Services"] = (
        (
            (average_annual_growth["Capital Input - Index"] / average_annual_growth["cap_lag"])
            ** (1 / (average_annual_growth["Year"] - average_annual_growth["year_lag"]))
        )
        - 1
    ) * 100
    average_annual_growth["Multifactor Productivity"] = (
        (
            (average_annual_growth["Total Factor Productivity"] / average_annual_growth["tfp_lag"])
            ** (1 / (average_annual_growth["Year"] - average_annual_growth["year_lag"]))
        )
        - 1
    ) * 100

    full_table = average_annual_growth[
        [
            "Year",
            "year_lag",
            "Real GDP",
            "Labor Input",
            "Capital Services",
            "Multifactor Productivity",
        ]
    ].reset_index()

    full_table["ColNames"] = (
        full_table["year_lag"].astype(int).astype(str) + "-" + full_table["Year"].astype(str)
    )
    full_table = full_table[
        ["ColNames", "Real GDP", "Labor Input", "Capital Services", "Multifactor Productivity"]
    ]

    full_table = full_table.merge(shares, how="left", on="ColNames").assign(
        real_gdp=lambda x: x["Real GDP"],
        lab_input=lambda x: x["Labor Share"] * x["Labor Input"],
        cap_services=lambda x: x["Capital Services"] * x["Capital Share"],
    )
    full_table["multifactor_prod"] = (
        full_table["real_gdp"] - full_table["lab_input"] - full_table["cap_services"]
    )

    avg_ann_growth = (
        full_table[
            ["ColNames", "Real GDP", "Labor Input", "Capital Services", "Multifactor Productivity"]
        ]
        .rename(
            columns={
                "Labor Input": "Labor input",
                "Capital Services": "Capital services",
                "Multifactor Productivity": "Multifactor productivity",
            }
        )
        .T
    )
    avg_ann_growth = avg_ann_growth.rename(columns=avg_ann_growth.iloc[0]).drop(
        avg_ann_growth.index[0]
    )
    avg_ann_growth["Difference"] = avg_ann_growth.iloc[:, 1] - avg_ann_growth.iloc[:, 0]

    contributions_to_growth = (
        full_table[["ColNames", "Real GDP", "lab_input", "cap_services", "multifactor_prod"]]
        .rename(
            columns={
                "lab_input": "Labor input",
                "cap_services": "Capital services",
                "multifactor_prod": "Multifactor productivity",
            }
        )
        .T
    )
    contributions_to_growth = contributions_to_growth.rename(
        columns=contributions_to_growth.iloc[0]
    ).drop(contributions_to_growth.index[0])
    contributions_to_growth["Difference"] = (
        contributions_to_growth.iloc[:, 1] - contributions_to_growth.iloc[:, 0]
    )

    # Combine for HTML
    trends_gr, trends_pp = avg_ann_growth.copy(), contributions_to_growth.copy()
    trends_gr = trends_gr / 100
    tbl_title = chart_config["title"]
    tbl1_html = (
        trends_gr.reset_index()
        .rename(
            {
                "index": "Component",
                f"{start_year}-{mid_year}": f"{start_year}–{mid_year}",
                f"{mid_year}-{end_year}": f"{mid_year}–{end_year}",
            },
            axis=1,
        )
        .style.format(
            {
                f"{start_year}–{mid_year}": _pd_format_precision("growth_rate_pct"),
                f"{mid_year}–{end_year}": _pd_format_precision("growth_rate_pct"),
                "Difference": _pd_format_precision("growth_rate_pct"),
            }
        )
        .set_table_attributes('class="table table-condensed"')
        .hide(axis="index")
        .to_html()
    )
    tbl1_html = tbl1_html.replace("<th id", '<th class="text-left" id', 1).replace(
        "<th id", '<th class="text-center" id'
    )
    inter_row = "Contributions to Growth<br>[Percentage points]<sup>1</sup>"
    footer = "\n".join(
        [
            '<tfoot><tr><td colspan="4"><ul class="footnote">',
            "\n".join(["<li>" + note + "</li>" for note in chart_config.get("notes")]),
            "</ul></td></tr></tfoot></table>",
        ]
    )
    tbl2_html = (
        trends_pp.reset_index()
        .rename({"index": "Component"}, axis=1)
        .style.format(precision=precision_rules["growth_rate_pct"], subset=pd.IndexSlice[[0], :])
        .format(
            precision=precision_rules["growth_rate_pct_contributions"],
            subset=pd.IndexSlice[[1, 2, 3], :],
        )
        .set_table_styles([{"selector": "thead", "props": [("display", "none")]}])
        .hide(axis="index")
        .to_html()
    )
    tbl_html = (
        "".join(tbl1_html.splitlines(keepends=True)[:-2])
        + f'<tr><td></td><td colspan=3 style="text-align: center">{inter_row}</tc></tr>\n'
        + "".join(tbl2_html.splitlines(keepends=True)[15:-1])
        + footer
    )
    tbl_html = (
        tbl_html.replace(
            "<thead>",
            "<thead>"
            + f'<tr><th>&nbsp;</th><th class="text-center" colspan="3">{tbl_title}</th></tr>',
        )
        .replace("%", '<span class="percent">%</span>')
        .replace(' col1"', ' col1" align="center"')
        .replace(' col2"', ' col2" align="center"')
        .replace(' col3"', ' col3" align="center"')
    )

    return tbl_html


def trade_balance(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    y_range=(0, 35),
    y_range2=(-8, 6),
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """Trade and the U.S. Economy Over Time: Total Trade and Trade Balances as Percentages of GDP

    Creates a graph of Total trades and trade balances (total, goods, services)
    as a percentage of GDP

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        y_range (tuple, optional): Primary y-axis range. Defaults to (0,35).
        y_range2 (tuple, optional): Secondary y-axis range. Defaults to (-8, 6).
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["Trade Balance"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    if end_year <= start_year:
        raise Exception("Need start_year<end_year.")

    gdp = get_data.nipa_series(
        **chart_config["GDP"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    total_exports = get_data.nipa_series(
        **chart_config["Total Exports"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    goods_exports = get_data.nipa_series(
        **chart_config["Goods Exports"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    services_exports = get_data.nipa_series(
        **chart_config["Services Exports"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    total_imports = get_data.nipa_series(
        **chart_config["Total Imports"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    goods_imports = get_data.nipa_series(
        **chart_config["Goods Imports"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    services_imports = get_data.nipa_series(
        **chart_config["Services Imports"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    net_exports_share = get_data.nipa_series(
        **chart_config["Net Exports GDP Share"], from_cache=from_cache, save_to_cache=save_to_cache
    )  # Scaled to match other shares
    if start_year < services_exports.index.min():
        raise Exception(f"start_year needs to be at least {services_exports.index.min()}")
    if end_year > services_exports.index.max():
        raise Exception(f"end_year needs to be at most {services_exports.index.max()}")

    goods_net_exports_share = (goods_exports - goods_imports) / gdp * 100  # Make numeral percent
    services_net_exports_share = (
        (services_exports - services_imports) / gdp * 100
    )  # Make numeral percent
    total_trade_share = (total_exports + total_imports) / gdp * 100  # Make numeral percent

    chart_df = pd.concat(
        [
            net_exports_share.rename("net_exports"),
            goods_net_exports_share.rename("goods"),
            services_net_exports_share.rename("services"),
            total_trade_share.rename("total_trade"),
        ],
        axis="columns",
    )
    _write_xlsx_sheet(
        pd.concat(
            [
                gdp.rename("gdp"),
                total_exports.rename("total_exports"),
                goods_exports.rename("goods_exports"),
                services_exports.rename("services_exports"),
                total_imports.rename("total_imports"),
                goods_imports.rename("goods_imports"),
                services_imports.rename("services_imports"),
                chart_df,
            ],
            axis="columns",
        ),
        "trade_balance",
    )
    chart_df = chart_df[start_year:end_year]
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)

    chart = bokeh.plotting.figure(
        y_axis_label="Total trade share of GDP, percent (bars)",
        y_range=y_range,
        x_range=(start_year - 1, end_year + 1),
        x_axis_type="datetime",
        **chart_config["size"],
    )
    chart.ygrid.grid_line_color = GRAY

    total_trade_bars = chart.vbar(
        "TimePeriod",
        top="total_trade",
        width=datetime.timedelta(days=250),
        color=GRAY,
        source=chart_data,
    )

    chart.extra_y_ranges["balances_shares"] = bokeh.models.Range1d(*y_range2)
    net_exports_line = chart.line(
        "TimePeriod",
        "net_exports",
        name="net_exports",
        y_range_name="balances_shares",
        color=DARK_GRAY,
        source=chart_data,
    )
    goods_line = chart.line(
        "TimePeriod",
        "goods",
        name="goods",
        y_range_name="balances_shares",
        color=BLUE,
        source=chart_data,
    )
    services_line = chart.line(
        "TimePeriod",
        "services",
        name="services",
        y_range_name="balances_shares",
        color=ORANGE,
        source=chart_data,
    )

    right_axis_title = "Trade balances, share of GDP, percent (lines)"

    chart.add_layout(
        bokeh.models.LinearAxis(
            y_range_name="balances_shares",
            axis_label=right_axis_title,
        ),
        place="right",
    )
    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("Total trade (exports + imports)", [total_trade_bars]),
                ("Trade balance", [net_exports_line]),
                ("Goods balance", [goods_line]),
                ("Services balance", [services_line]),
            ]
        ),
        place="below",
    )

    tt1 = [
        ("Year", "@TimePeriod{%Y}"),
        ("Percent", "@total_trade{0,0." + "0" * precision_rules["shares"] + "}"),
    ]
    tt1 = _custom_tooltip(tt1)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[total_trade_bars],
            tooltips=tt1,
        )
    )
    tt2 = [
        ("Year", "@TimePeriod{%Y}"),
        ("Percent", "@$name{0,0." + "0" * precision_rules["shares"] + "}"),
    ]
    tt2 = _custom_tooltip(tt2)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[net_exports_line, goods_line, services_line],
            tooltips=tt2,
        )
    )

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def financing_trade(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    y_range=(-16, 16),
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """Financing International Trade: Foreign Debt Service and Debt

    Creates a graph of ratios of foreign debt service (to exports and to income)
    and the ratio of international debt to assets.

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        y_range (tuple, optional): Y-axis range. Defaults to (-16, 16).
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["Financing Trade"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    if end_year <= start_year:
        raise Exception("Need start_year<end_year.")

    household_assets = get_data.fred_series(
        **chart_config["Household Assets"], from_cache=from_cache, save_to_cache=save_to_cache
    )

    exports = get_data.nipa_series(
        **chart_config["Exports"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    gdi = get_data.nipa_series(
        **chart_config["GDI"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    investment_receipts = get_data.ita_series(
        **chart_config["Investment Income Receipts from ROW"],
        from_cache=from_cache,
        save_to_cache=save_to_cache,
    )
    investment_payments = get_data.ita_series(
        **chart_config["Investment Income Payments to ROW"],
        from_cache=from_cache,
        save_to_cache=save_to_cache,
    )
    net_investment = get_data.fred_series(
        **chart_config["US Net International Investment Position"],
        from_cache=from_cache,
        save_to_cache=save_to_cache,
    )
    if start_year < investment_payments.index.min():
        raise Exception(f"start_year needs to be at least {investment_payments.index.min()}")
    if end_year > investment_payments.index.max():
        raise Exception(f"end_year needs to be at most {investment_payments.index.max()}")

    debt_service_to_exports = (
        (investment_payments - investment_receipts) / exports * 100
    )  # Make numeral percent
    debt_service_to_income = (
        (investment_payments - investment_receipts) / gdi * 100
    )  # Make numeral percent
    debt_to_assets = -net_investment / household_assets * 100  # Make numeral percent

    chart_df = pd.concat(
        [
            debt_service_to_exports.rename("debt_service_to_exports"),
            debt_service_to_income.rename("debt_service_to_income"),
            debt_to_assets.rename("debt_to_assets"),
        ],
        axis="columns",
    )[start_year:end_year]
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)
    _write_xlsx_sheet(
        pd.concat(
            [
                household_assets.rename("household_assets"),
                exports.rename("exports"),
                gdi.rename("gdi"),
                investment_receipts.rename("investment_receipts"),
                investment_payments.rename("investment_payments"),
                net_investment.rename("net_investment"),
                chart_df,
            ],
            axis="columns",
        )[start_year:end_year],
        "financing_trade",
    )

    chart = bokeh.plotting.figure(
        y_axis_label="Percent",
        y_range=y_range,
        x_range=(start_year, end_year),
        x_axis_type="datetime",
        **chart_config["size"],
    )
    chart.ygrid.grid_line_color = GRAY
    chart.xaxis.axis_line_color = None
    chart.yaxis.axis_line_color = None
    chart.outline_line_color = None

    debt_service_to_exports_line = chart.line(
        "TimePeriod",
        "debt_service_to_exports",
        name="debt_service_to_exports",
        color=BLUE,
        source=chart_data,
    )
    debt_service_to_income_line = chart.line(
        "TimePeriod",
        "debt_service_to_income",
        name="debt_service_to_income",
        color=ORANGE,
        source=chart_data,
    )
    debt_to_assets_line = chart.line(
        "TimePeriod", "debt_to_assets", name="debt_to_assets", color=DARK_GRAY, source=chart_data
    )

    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("Intl debt service to exports ratio", [debt_service_to_exports_line]),
                ("Intl debt service to income ratio", [debt_service_to_income_line]),
                ("Intl debt to asset ratio", [debt_to_assets_line]),
            ]
        ),
        place="below",
    )

    tt = [
        ("Year", "@TimePeriod{%Y}"),
        ("Percent", "@$name{0." + "0" * precision_rules["shares"] + "}"),
    ]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[
                debt_service_to_exports_line,
                debt_service_to_income_line,
                debt_to_assets_line,
            ],
            tooltips=tt,
        )
    )

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def mne_employment(
    start_year: Union[int, pd.Series] = None,
    end_year: Union[int, pd.Series] = None,
    y_range=(0, 30000),
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """U.S. Multinational Companies' Employment in U.S. and Foreign Subsidiaries (chart)

    Creates a chart of domestic and foreign employment of US Multinational companies.

    Args:
        start_year (Union[int,pd.Series], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,pd.Series], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        y_range (tuple, optional): Y-axis range. Defaults to (0, 30000).
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["MNE Employment"]
    start_year = start_year or chart_config["defaults"]["start_year"]
    end_year = end_year or chart_config["defaults"]["end_year"]
    if end_year <= start_year:
        raise Exception("Need start_year<end_year.")
    if start_year < 1999:  # HARDCODE
        raise Exception("start_year needs to be at least 1999")
    # First fix the top
    if end_year > 2014:
        years = [1999, 2004, 2009] + list(range(2014, end_year + 1))
    else:
        years = [year for year in [1999, 2004, 2009, 2014] if year <= end_year]
    # Next fix the bottom
    years = [year for year in years if year >= start_year]

    pre08_dom_employ = get_data.mne_series(
        **chart_config["Pre-2008 Domestic Data"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    pre08_dom_employ = pre08_dom_employ.loc[
        (pre08_dom_employ["RowCode"] == "0000") & (pre08_dom_employ["Year"].isin(years))
    ][["Year", "SeriesName", "DataValueUnformatted"]].rename(
        columns={"DataValueUnformatted": "DomesticEmploy"}
    )

    post08_dom_employ = get_data.mne_series(
        **chart_config["Post-2008 Domestic Data"],
        from_cache=from_cache,
        save_to_cache=save_to_cache,
    )
    if end_year > post08_dom_employ["Year"].max():
        raise Exception(f"end_year needs to be at most {post08_dom_employ['Year'].max()}")
    post08_dom_employ = post08_dom_employ.loc[
        (post08_dom_employ["RowCode"] == "0000") & (post08_dom_employ["Year"].isin(years))
    ][["Year", "SeriesName", "DataValueUnformatted"]].rename(
        columns={"DataValueUnformatted": "DomesticEmploy"}
    )

    pre08_for_employ = get_data.mne_series(
        **chart_config["Pre-2008 Foreign Data"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    pre08_for_employ = pre08_for_employ.loc[
        (pre08_for_employ["ColumnCode"] == "0000")
        & (pre08_for_employ["RowCode"] == "000")
        & (pre08_for_employ["Year"].isin(years))
    ][["Year", "SeriesName", "DataValueUnformatted"]].rename(
        columns={"DataValueUnformatted": "ForeignEmploy"}
    )

    post08_for_employ = get_data.mne_series(
        **chart_config["Post-2008 Foreign Data"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    post08_for_employ = post08_for_employ.loc[
        (post08_for_employ["ColumnCode"] == "0000")
        & (post08_for_employ["RowCode"] == "000")
        & (post08_for_employ["Year"].isin(years))
    ][["Year", "SeriesName", "DataValueUnformatted"]].rename(
        columns={"DataValueUnformatted": "ForeignEmploy"}
    )

    foreign_employ = pd.concat([pre08_for_employ, post08_for_employ])
    domestic_employ = pd.concat([pre08_dom_employ, post08_dom_employ])

    mfn_employ = domestic_employ.merge(foreign_employ, on=["Year", "SeriesName"])
    mfn_employ["DomesticEmploy"] = mfn_employ["DomesticEmploy"].astype(float)
    mfn_employ["ForeignEmploy"] = mfn_employ["ForeignEmploy"].astype(float)

    chart_data = bokeh.plotting.ColumnDataSource(mfn_employ)
    _write_xlsx_sheet(mfn_employ, "mne_employment")

    chart = bokeh.plotting.figure(
        y_range=y_range, y_axis_label="Thousands of employees", **chart_config["size"]
    )
    chart.ygrid.grid_line_color = GRAY

    domestic = chart.vbar(
        x=bokeh.transform.dodge("Year", -0.22, range=chart.x_range),
        top="DomesticEmploy",
        source=chart_data,
        width=0.4,
        color=BLUE,
    )

    foreign = chart.vbar(
        x=bokeh.transform.dodge("Year", 0.22, range=chart.x_range),
        top="ForeignEmploy",
        source=chart_data,
        width=0.4,
        color=ORANGE,
    )

    tt1 = [("Year", "@Year"), ("Domestic Employment", "@DomesticEmploy{0,0}")]
    tt1 = _custom_tooltip(tt1)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[domestic],
            tooltips=tt1,
        ),
    )
    tt2 = [("Year", "@Year"), ("Foreign Employment", "@ForeignEmploy{0,0}")]
    tt2 = _custom_tooltip(tt2)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[foreign],
            tooltips=tt2,
        )
    )

    chart.add_layout(
        bokeh.models.Legend(
            items=[("Domestic employment", [domestic]), ("Foreign employment", [foreign])]
        ),
        place="below",
    )
    chart.xaxis.ticker = years
    chart.xaxis.major_label_orientation = 0.75

    layout = _add_metatext(chart=chart, chart_config=chart_config, end_year=end_year)

    return layout


def foreign_subsidiary_sales_table(
    year: Union[int, str, pd.Period] = None, from_cache: bool = False, save_to_cache: bool = True
) -> str:
    """U.S. Multinational Companies' Employment in U.S. and Foreign Subsidiaries (table 1)

    Creates a table of foreign subsidiary sales

    Args:
        year (Union[int,str,pd.Period], optional): Year. Defaults to None,
          which uses the value in chart_config.yml.
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        str: HTML table
    """
    chart_config = config["Foreign Sales"]
    year = year or chart_config["defaults"]["year"]
    if year < 2009:  # HARDCODE
        raise Exception("Need year>=2009")
    cols = ["To the United States", "To the host country", "To other foreign countries"]

    try:
        goods_supplied = get_data.mne_series(
            year=year,
            **chart_config["Goods Supplied"],
            from_cache=from_cache,
            save_to_cache=save_to_cache,
        )
        services_supplied = get_data.mne_series(
            year=year,
            **chart_config["Services Supplied"],
            from_cache=from_cache,
            save_to_cache=save_to_cache,
        )
    except Exception as e:
        if e.args[0][:13] == "Error Code: 1":  # Error Code: 1. Description: no data found "
            raise Exception("Invalid year. Maybe there's no data for that year")
        else:
            raise e

    _write_xlsx_sheet(
        pd.concat([goods_supplied, services_supplied], axis="columns"),
        "foreign_subsidiary_sales_table",
    )
    goods_supplied = goods_supplied.loc[
        (goods_supplied["ColumnCode"] == "000") & (goods_supplied["ColumnParent"].isin(cols))
    ]
    services_supplied = services_supplied.loc[
        (services_supplied["ColumnCode"] == "000") & (services_supplied["ColumnParent"].isin(cols))
    ]

    foreign_subsidiary_sales = pd.DataFrame(
        pd.concat([goods_supplied, services_supplied])
        .astype({"DataValueUnformatted": "int"})
        .groupby("ColumnParent")["DataValueUnformatted"]
        .sum()
    )
    foreign_subsidiary_sales["shareTotal"] = (
        foreign_subsidiary_sales.DataValueUnformatted
        / foreign_subsidiary_sales.DataValueUnformatted.sum()
    ) * 100
    foreign_subsidiary_sales.loc["Total"] = foreign_subsidiary_sales.sum()
    foreign_subsidiary_sales = (
        foreign_subsidiary_sales.rename(
            columns={
                "ColumnParent": "Destination",
                "DataValueUnformatted": "ForeignSubsidiarySales",
            }
        )
        .sort_values(by="ForeignSubsidiarySales", ascending=False)
        .reset_index()
    )
    foreign_subsidiary_sales = foreign_subsidiary_sales.reset_index()[
        ["ColumnParent", "shareTotal"]
    ]

    # Format for HTML
    fss_df = foreign_subsidiary_sales.copy()
    fss_df["shareTotal"] = fss_df["shareTotal"] / 100
    fss_df = fss_df.drop([0], axis=0).rename(
        {"ColumnParent": "Total", "shareTotal": "100%"}, axis=1
    )
    tbl_title = chart_config["table"]["title"].format(year=year)
    tbl_html = (
        fss_df.style.set_table_attributes('class="table table-condensed"')
        .format({"100%": "{:.0%}"})
        .hide(axis="index")
        .set_caption(tbl_title)
        .to_html()
    )
    footer = (
        '<tfoot><tr><td colspan="2"><ul>'
        + "<li>"
        + chart_config["table"]["note"]
        + "</li>"
        + "</ul></td></tr></tfoot>"
    )
    tbl_html = (
        tbl_html.replace(' col1"', ' col1" align="center"')
        .replace("</table>", footer + "</table>")
        .replace("%", '<span class="percent">%</span>')
    )
    return tbl_html


def foreign_subsidiary_employment(
    years=None, from_cache: bool = False, save_to_cache: bool = True
) -> str:
    """U.S. Multinational Companies' Employment in U.S. and Foreign Subsidiaries (table 2)

    Creates a table of share of foreign subsidary employment by level of country income.

    Notes:
    Foreign subsidary employment data for some countries is unavailable or has been suppressed, thus data
    from these countries are not included in the income-level sums. They are still represented in the yearly
    total as it is reported directly by BEA.

    For years prior to 1999, the "International" category consists of affiliates that have operations spanning
    more than one country and that are engaged in petroleum shipping, other water transportation, or offshore
    oil and gas drilling. These are also not included in the income-level sums, but they are represented
    in the yearly total.

    For a given year, shares may not sum to 100 percent due to rounding and the aforementioned discrepancies.

    Args:
        years (list, optional): Years. Defaults to [1997,2000,2007,2019,2020].
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        str: HTML table
    """
    chart_config = config["Foreign Employment"]

    if years is None:
        years = [int(year_str.strip()) for year_str in chart_config["defaults"]["years"].split(",")]
    if min(years) < 1997:  # HARDCODE
        raise Exception("Minimum year is 1997")

    full_file = get_data.cache_excel_file(
        "https://datacatalogfiles.worldbank.org/ddh-published/0037712/DR0090754/OGHIST.xlsx",
        from_cache,
        save_to_cache,
    )

    excel_df = pd.read_excel(
        io=full_file,
        sheet_name="Country Analytical History",
        index_col=0,
        header=0,
        usecols="B,C:AL",
        skiprows=lambda x: x not in range(5, 229),  # remove header
        dtype=str,
    )
    excel_df = excel_df.drop(excel_df.index[:5]).replace(  # remove sub-header
        {"L": "LIC", "LM": "MIC", "UM": "MIC", "H": "HIC"}
    )
    excel_df.index.names = ["Country"]
    excel_df.reset_index(inplace=True)

    # align country names between the two datasources
    replacement_dict = {
        "Bahamas, The": "Bahamas",
        "Brunei Darussalam": "Brunei",
        "Congo, Dem. Rep.": "Congo (Kinshasa)",
        "Congo, Rep.": "Congo (Brazzaville)",
        "Côte d'Ivoire": "Cote D'Ivoire",
        "Curaçao": "Curacao",
        "Egypt, Arab Rep.": "Egypt",
        "Micronesia, Fed. Sts.": "Federated States of Micronesia",
        "Hong Kong SAR, China": "Hong Kong",
        "Korea, Rep.": "South Korea",
        "Kyrgyz Republic": "Kyrgyzstan",
        "Lao PDR": "Laos",
        "Macao SAR, China": "Macau",
        "North Macedonia": "Macedonia",
        "Marshall Islands": "Republic of Marshall Islands",
        "Russian Federation": "Russia",
        "São Tomé and Príncipe": "Sao Tome and Principe",
        "Sint Maarten (Dutch part)": "Sint Maarten",
        "Slovak Republic": "Slovakia",
        "Eswatini": "Swaziland",
        "Syrian Arab Republic": "Syria",
        "Taiwan, China": "Taiwan",
        "Türkiye": "Turkey",
        "Venezuela, RB": "Venezuela",
        "Yemen, Rep.": "Yemen",
    }
    income_level_df = pd.melt(
        excel_df, id_vars="Country", var_name="Year", value_name="incomeLevel"
    ).replace(replacement_dict)

    pre08_employ = get_data.mne_series(
        **chart_config["Pre-2008 Foreign Subsidiary Employment Data"],
        from_cache=from_cache,
        save_to_cache=save_to_cache,
    )

    post08_employ = get_data.mne_series(
        **chart_config["Post-2008 Foreign Subsidiary Employment Data"],
        from_cache=from_cache,
        save_to_cache=save_to_cache,
    )
    max_year = max(years)
    if max_year > post08_employ["Year"].max():
        raise Exception(f"max(years) needs to be at most {post08_employ['Year'].max()}")

    mofa_employ = pd.concat([pre08_employ, post08_employ])

    mofa_employ_w_incLevel = mofa_employ.merge(
        income_level_df, how="left", left_on=["Year", "Row"], right_on=["Year", "Country"]
    )
    _write_xlsx_sheet(mofa_employ_w_incLevel, "foreign_subsidiary_employment")

    table_data = mofa_employ_w_incLevel[
        pd.to_numeric(mofa_employ_w_incLevel["DataValueUnformatted"], errors="coerce").notnull()
    ].copy()

    table_data["Employees"] = table_data["DataValueUnformatted"].astype("float")
    table_data["IncomeLevelEmployees"] = table_data.groupby(["Year", "incomeLevel"])[
        "Employees"
    ].transform("sum")
    table_data["TotalEmployees"] = table_data.groupby(["Year"])["Employees"].transform("max")
    table_data["ShareOfMOFAEmploy"] = (
        table_data["IncomeLevelEmployees"] / table_data["TotalEmployees"]
    )

    collapsed_df = (
        table_data.groupby(["Year", "incomeLevel"])
        .agg({"IncomeLevelEmployees": "mean", "TotalEmployees": "mean"})
        .reset_index()
    )
    collapsed_df["ShareOfMOFAEmploy"] = (
        collapsed_df["IncomeLevelEmployees"] / collapsed_df["TotalEmployees"]
    ) * 100
    mofa_table = (
        collapsed_df[collapsed_df["Year"].isin(years)]
        .pivot_table("ShareOfMOFAEmploy", ["Year"], "incomeLevel")
        .reindex(["HIC", "MIC", "LIC"], axis=1)
        .rename(columns={"HIC": "High Income", "MIC": "Middle Income", "LIC": "Low Income"})
    )

    # Format for HTML
    fse_tbl = mofa_table.copy()
    fse_tbl = fse_tbl.reset_index().rename(
        {"Year": "Income", "High Income": "High", "Middle Income": "Middle", "Low Income": "Low"},
        axis=1,
    )
    fse_tbl["High"] = fse_tbl["High"] / 100
    fse_tbl["Middle"] = fse_tbl["Middle"] / 100
    fse_tbl["Low"] = fse_tbl["Low"] / 100
    fse_tbl_html = (
        fse_tbl.style.format({"High": "{:.0%}", "Middle": "{:.0%}", "Low": "{:.0%}"})
        .set_table_attributes('class="table table-condensed"')
        .set_caption(chart_config["table"]["title"])
        .hide(axis="index")
        .to_html()
    )

    footer = (
        '<tfoot><tr><td colspan="4"><ul>'
        + "\n".join(["<li>" + note + "</li>" for note in chart_config["table"].get("notes")])
        + "</ul></td></tr></tfoot>"
    )
    fse_tbl_html = (
        fse_tbl_html.replace("%", '<span class="percent">%</span>')
        .replace(' col0"', ' col0" align="left"')
        .replace(' col1"', ' col1" align="center"')
        .replace(' col2"', ' col2" align="center"')
        .replace(' col3"', ' col3" align="center"')
        .replace("</table>", footer + "</table>")
    )
    return fse_tbl_html


def federal_budget(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """U.S. Budget and the Economy Over Time: Federal Surpluses and Deficits as a Percentage of GDP

    Creates a graph of budget surpluses and deficits.

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["Federal Budget"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    if end_year <= start_year:
        raise Exception("Need start_year<end_year.")

    deficits = get_data.fred_series(
        **chart_config["Deficit"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    nominal_gdp = pd.DataFrame(
        get_data.nipa_series(
            **chart_config["Nominal GDP"], from_cache=from_cache, save_to_cache=save_to_cache
        )
    ).rename(columns={"DataValue": "nom_gdp"})
    if start_year < deficits.index.min():
        raise Exception(f"start_year needs to be at least {deficits.index.min()}")
    if end_year > deficits.index.max():
        raise Exception(f"end_year needs to be at most {deficits.index.max()}")
    chart_df = nominal_gdp

    chart_df["gdp_fy"] = np.where(
        nominal_gdp.index.year > 1976,
        (nominal_gdp["nom_gdp"].shift(1) * (1 / 3)) + (nominal_gdp["nom_gdp"] * (2 / 3)),
        (nominal_gdp["nom_gdp"] + nominal_gdp["nom_gdp"].shift(1)) / 2,
    )
    chart_df["deficits"] = deficits
    chart_df["percentage"] = (chart_df["deficits"] / chart_df["gdp_fy"]) * 100

    chart_data = bokeh.plotting.ColumnDataSource(
        chart_df.assign(bar_color=lambda x: np.where(x.percentage >= 0, ORANGE, BLUE))
    )
    _write_xlsx_sheet(chart_df, "federal_budget")

    chart = bokeh.plotting.figure(
        y_axis_label="Percent",
        x_range=(start_year - 1, end_year + 1),
        x_axis_type="datetime",
        **chart_config["size"],
    )

    bar_width = 250
    dodge_amount = bar_width / 2 * 24 * 60 * 60 * 1000
    deficit_bars = chart.vbar(
        x=bokeh.transform.dodge("TimePeriod", dodge_amount),
        top="percentage",
        width=datetime.timedelta(days=bar_width),
        source=chart_data,
        color="bar_color",
        line_color="white",
    )

    tt = [
        ("Year", "@TimePeriod{%Y}"),
        ("Surplus/Deficit Percent", "@percentage{0." + "0" * precision_rules["shares"] + "}"),
    ]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[deficit_bars],
            tooltips=tt,
        )
    )

    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("Surplus", [chart.vbar(color=ORANGE)]),
                ("Deficit", [chart.vbar(color=BLUE)]),
                ("Recessions", [chart.vbar(color=GRAY)]),
            ]
        ),
        place="below",
    )
    for recession in _recession_shading(
        start_year, end_year, from_cache=from_cache, save_to_cache=save_to_cache
    ):
        chart.add_layout(recession)

    chart.xaxis.ticker = _year_ticks(start_year, end_year, gap=10)
    chart.ygrid.grid_line_color = GRAY
    chart.xaxis.axis_line_color = None
    chart.yaxis.axis_line_color = None
    chart.outline_line_color = None

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def budget_deficit(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    y_range=(-8, 20),
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """U.S. Budget Deficits: Federal Government Debt Service and Debt

    Creates a graph of federal net borrow to income and federal debt service ratio.

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        y_range (tuple, optional): Y-axis range. Defaults to (-8, 20).
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["Budget Deficit"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    if end_year <= start_year:
        raise Exception("Need start_year<end_year.")

    income = get_data.nipa_series(
        **chart_config["National Income"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    receipts = get_data.nipa_series(
        **chart_config["Federal Receipts"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    expenditures = get_data.nipa_series(
        **chart_config["Federal Expenditures"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    gni = get_data.nipa_series(
        **chart_config["GNI"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    interest = get_data.nipa_series(
        **chart_config["Federal Interest Payments"],
        from_cache=from_cache,
        save_to_cache=save_to_cache,
    )
    if start_year < interest.index.min():
        raise Exception(f"start_year needs to be at least {interest.index.min()}")
    if end_year > interest.index.max():
        raise Exception(f"end_year needs to be at most {interest.index.max()}")

    borrowing_to_income = (expenditures - receipts) / income * 100  # Make numeral percent
    debt_service_ratio = interest / gni * 100  # Make numeral percent

    chart_df = pd.concat(
        [
            borrowing_to_income.rename("borrowing_to_income"),
            debt_service_ratio.rename("debt_service_ratio"),
        ],
        axis="columns",
    )
    _write_xlsx_sheet(
        pd.concat(
            [
                income.rename("income"),
                receipts.rename("receipts"),
                expenditures.rename("expenditures"),
                gni.rename("gni"),
                interest.rename("interest"),
                chart_df,
            ],
            axis="columns",
        ),
        "budget_deficit",
    )
    chart_df = chart_df[start_year:end_year]
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)

    chart = bokeh.plotting.figure(
        y_axis_label="Percent",
        y_range=y_range,
        x_range=(start_year, end_year),
        x_axis_type="datetime",
        **chart_config["size"],
    )
    chart.ygrid.grid_line_color = GRAY
    chart.xaxis.axis_line_color = None
    chart.yaxis.axis_line_color = None
    chart.outline_line_color = None

    borrowing_line = chart.line(
        "TimePeriod",
        "borrowing_to_income",
        name="borrowing_to_income",
        color=ORANGE,
        source=chart_data,
    )
    debt_service_line = chart.line(
        "TimePeriod", "debt_service_ratio", name="debt_service_ratio", color=BLUE, source=chart_data
    )

    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("Federal net borrowing to income ratio", [borrowing_line]),
                ("Federal debt service ratio", [debt_service_line]),
                ("Recessions", [chart.vbar(color=GRAY)]),
            ]
        ),
        place="below",
    )
    for recession in _recession_shading(
        start_year, end_year, from_cache=from_cache, save_to_cache=save_to_cache
    ):
        chart.add_layout(recession)

    chart.xaxis.ticker = _year_ticks(start_year, end_year, gap=4)

    tt = [
        ("Year", "@TimePeriod{%Y}"),
        ("Percent", "@$name{0." + "0" * precision_rules["shares"] + "}"),
    ]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[borrowing_line, debt_service_line],
            tooltips=tt,
        )
    )

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def saving_investment(
    start_year: Union[int, str, pd.Period] = None,
    end_year: Union[int, str, pd.Period] = None,
    y_range=(-10, 20),
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """Saving & Investment for the Future: Net Saving and Net Investment as Percentages of GDP

    Creates a graph of net savings rate and net investment rate

    Args:
        start_year (Union[int,str,pd.Period], optional): Start year. Defaults to None,
          which uses the value in chart_config.yml.
        end_year (Union[int,str,pd.Period], optional): End year. Defaults to None,
          which uses the value in chart_config.yml.
        y_range (tuple, optional): Y-axis range. Defaults to (-10, 20).
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["Saving and Investment"]
    start_year = pd.Period(start_year or chart_config["defaults"]["start_year"])
    end_year = pd.Period(end_year or chart_config["defaults"]["end_year"])
    if end_year <= start_year:
        raise Exception("Need start_year<end_year.")

    saving = get_data.nipa_series(
        **chart_config["Saving"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    investment = get_data.nipa_series(
        **chart_config["Investment"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    gdp = get_data.nipa_series(
        **chart_config["GDP"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    if start_year < investment.index.min():
        raise Exception(f"start_year needs to be at least {investment.index.min()}")
    if end_year > investment.index.max():
        raise Exception(f"end_year needs to be at most {investment.index.max()}")

    saving_rate = saving / gdp * 100  # Make numeral percent
    investment_rate = investment / gdp * 100  # Make numeral percent

    chart_df = pd.concat(
        [
            saving_rate.rename("saving_rate"),
            investment_rate.rename("investment_rate"),
        ],
        axis="columns",
    )
    _write_xlsx_sheet(pd.concat([gdp.rename("gdp"), chart_df], axis="columns"), "saving_investment")
    chart_df = chart_df[start_year:end_year]
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)

    chart = bokeh.plotting.figure(
        y_axis_label="Percent",
        y_range=y_range,
        x_range=(start_year, end_year),
        x_axis_type="datetime",
        **chart_config["size"],
    )
    chart.ygrid.grid_line_color = GRAY
    chart.xaxis.axis_line_color = "black"
    chart.yaxis.axis_line_color = None
    chart.outline_line_color = None

    saving_line = chart.line(
        "TimePeriod", "saving_rate", name="saving_rate", color=BLUE, source=chart_data
    )
    investment_line = chart.line(
        "TimePeriod", "investment_rate", name="investment_rate", color=ORANGE, source=chart_data
    )

    chart.add_layout(
        bokeh.models.Legend(
            items=[
                ("Net saving rate", [saving_line]),
                ("Net investment rate", [investment_line]),
                ("Recessions", [chart.vbar(color=GRAY)]),
            ]
        ),
        place="below",
    )
    for recession in _recession_shading(
        start_year, end_year, from_cache=from_cache, save_to_cache=save_to_cache
    ):
        chart.add_layout(recession)

    chart.xaxis.ticker = _year_ticks(start_year, end_year, gap=10)

    tt = [
        ("Year", "@TimePeriod{%Y}"),
        ("Percent", "@$name{0." + "0" * precision_rules["shares"] + "}"),
    ]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[saving_line, investment_line],
            tooltips=tt,
        )
    )

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_year=start_year, end_year=end_year
    )

    return layout


def business_cycles(
    start_quarter: Union[int, str, pd.Period] = None,
    end_quarter: Union[int, str, pd.Period] = None,
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> bokeh.layouts.Column:
    """Trends in U.S. Business Cycles: Rates of Change and Duration (chart)

    Creates a graph of real GDP highlighting the most recent contraction and expansion.

    Args:
        start_quarter (Union[int,str,pd.Period], optional): Start quarter. Defaults to None,
          which uses the value in chart_config.yml.
        end_quarter (Union[int,str,pd.Period], optional): End quarter. Defaults to None,
          which uses the value in chart_config.yml.
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        bokeh.layouts.Column: Column layout of the graph and metadata
    """
    chart_config = config["Business Cycles"]
    start_quarter = pd.Period(start_quarter or chart_config["defaults"]["start_quarter"])
    end_quarter = pd.Period(end_quarter or chart_config["defaults"]["end_quarter"])
    if end_quarter <= start_quarter:
        raise Exception("Need start_quarter<end_quarter.")

    real_gdp = (
        get_data.nipa_series(
            **chart_config["Real GDP"], from_cache=from_cache, save_to_cache=save_to_cache
        )
        / 1000
    )  # scale to billions of dollars
    all_cycles = get_data.business_cycles(from_cache=from_cache, save_to_cache=save_to_cache)
    gdp_quantity = get_data.nipa_series(
        **chart_config["GDP Quantity Index"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    if start_quarter < gdp_quantity.index.min():
        raise Exception(f"start_quarter needs to be at least {gdp_quantity.index.min()}")
    if end_quarter > gdp_quantity.index.max():
        raise Exception(f"end_quarter needs to be at most {gdp_quantity.index.max()}")

    cycles = all_cycles[start_quarter:end_quarter]
    expansions, contractions = [], []
    for (start, peak), (end, _) in zip(cycles.items(), cycles[1:].items()):
        (contractions if peak else expansions).append(
            {
                "dates": f'{start.strftime("%YQ%q")}-{end.strftime("%YQ%q")}',
                "length": (end - start).n,
                "mean_pct_change": (
                    (gdp_quantity[end] / gdp_quantity[start]) ** (4 / (end - start).n) - 1
                ),
            }
        )

    chart_df = pd.DataFrame(real_gdp.rename("gdp")[start_quarter:end_quarter])
    chart_df["TimePeriod_str"] = chart_df.index.to_series().astype(str)
    chart_data = bokeh.plotting.ColumnDataSource(chart_df)
    _write_xlsx_sheet(chart_df, "business_cycles")

    chart = bokeh.plotting.figure(
        y_axis_label=chart_config["y_axis_label"],
        x_range=(start_quarter, end_quarter),
        x_axis_type="datetime",
        **chart_config["size"],
    )
    chart.y_range.start = 0
    chart.ygrid.grid_line_color = GRAY
    chart.yaxis.formatter = bokeh.models.NumeralTickFormatter(format="0,0")

    gdp_line = chart.line("TimePeriod", "gdp", name="gdp", color=BLUE, source=chart_data)

    contraction_shading = [
        BoxAnnotation(
            left=pd.Period(c["dates"].split("-")[0]),
            right=pd.Period(c["dates"].split("-")[1]),
            fill_color=GRAY,
            level="underlay",
            line_width=0,
        )
        for c in contractions
    ]
    for contraction in contraction_shading:
        chart.add_layout(contraction)

    last_expansion = {
        "start": pd.Period(expansions[-1]["dates"].split("-")[0]),
        "end": pd.Period(expansions[-1]["dates"].split("-")[1]),
        "length": expansions[-1]["length"],
    }
    chart.add_layout(
        bokeh.models.Arrow(
            start=bokeh.models.TeeHead(line_color=BLUE, line_width=1, size=10),
            end=bokeh.models.TeeHead(line_color=BLUE, line_width=1, size=10),
            line_width=1,
            line_color=BLUE,
            x_start=last_expansion["start"],
            x_end=last_expansion["end"],
            y_start=real_gdp[last_expansion["start"]] - 500,
            y_end=real_gdp[last_expansion["start"]] - 500,
        )
    )
    chart.add_layout(
        bokeh.models.Label(
            x=last_expansion["start"]
            + int((last_expansion["end"] - last_expansion["start"]).n / 2),
            y=real_gdp[last_expansion["start"]] - 1_000,
            text=f'Previous\nexpansion:\n{last_expansion["length"]} qtrs',
            text_baseline="top",
        )
    )

    legend_items = [("Real GDP", [gdp_line]), ("Contractions", [chart.vbar(color=GRAY)])]

    tt = [
        ("Quarter", "@TimePeriod_str"),
        ("Value", "@$name{$0,0." + "0" * precision_rules["levels_billions"] + "}"),
    ]
    tt = _custom_tooltip(tt)
    chart.add_tools(
        bokeh.models.HoverTool(
            renderers=[gdp_line],
            tooltips=tt,
        )
    )
    current = {"start": all_cycles.index[-1], "type": "contraction" if cycles[-1] else "expansion"}
    if current["start"] < end_quarter:
        chart.add_layout(
            BoxAnnotation(
                left=pd.Period(current["start"]),
                fill_color=ORANGE,
                fill_alpha=0.5,
                level="underlay",
                line_width=0,
            )
        )
        legend_items.append((f'Current {current["type"]}', [chart.vbar(color=ORANGE, alpha=0.5)]))

    chart.add_layout(bokeh.models.Legend(items=legend_items), place="below")

    layout = _add_metatext(
        chart=chart, chart_config=chart_config, start_quarter=start_quarter, end_quarter=end_quarter
    )

    return layout


def business_cycle_table(
    start_quarter: Union[int, str, pd.Period] = None,
    end_quarter: Union[int, str, pd.Period] = None,
    from_cache: bool = False,
    save_to_cache: bool = True,
) -> str:
    """Trends in U.S. Business Cycles: Rates of Change and Duration (table)

    Creates a table of contractions/expansions with their average change in GDP.

    Args:
        start_quarter (Union[int,str,pd.Period], optional): Start quarter. Defaults to None,
          which uses the value in chart_config.yml.
        end_quarter (Union[int,str,pd.Period], optional): End quarter. Defaults to None,
          which uses the value in chart_config.yml.
        from_cache (bool, optional): Whether to retrieve from cache or not
          (and then pull data from APIs). Defaults to False.
        save_to_cache (bool, optional): If pulling from APIs, should we save data to the cache.
          Defaults to True.

    Returns:
        str: Column layout of the graph and metadata
    """
    chart_config = config["Business Cycle Table"]
    start_quarter = pd.Period(start_quarter or chart_config["defaults"]["start_quarter"])
    end_quarter = pd.Period(end_quarter or chart_config["defaults"]["end_quarter"])
    if end_quarter <= start_quarter:
        raise Exception("Need start_quarter<end_quarter.")

    all_cycles = get_data.business_cycles(from_cache=from_cache, save_to_cache=save_to_cache)
    gdp_quantity = get_data.nipa_series(
        **chart_config["GDP Quantity Index"], from_cache=from_cache, save_to_cache=save_to_cache
    )
    if start_quarter < gdp_quantity.index.min():
        raise Exception(f"start_quarter needs to be at least {gdp_quantity.index.min()}")
    if end_quarter > gdp_quantity.index.max():
        raise Exception(f"end_quarter needs to be at most {gdp_quantity.index.max()}")

    if end_quarter > all_cycles.index[-1]:
        all_cycles = pd.concat([all_cycles, pd.Series({end_quarter: not all_cycles[-1]})])

    cycles = all_cycles[min(gdp_quantity.index) :]
    expansions, contractions = [], []
    for (start, peak), (end, _) in zip(cycles.items(), cycles[1:].items()):
        if start_quarter <= start <= end_quarter:
            (contractions if peak else expansions).append(
                {
                    "dates": f'{start.strftime("%YQ%q")}–{end.strftime("%YQ%q")}',
                    "length": (end - start).n,
                    "mean_pct_change": (
                        (gdp_quantity[end] / gdp_quantity[start]) ** (4 / (end - start).n) - 1
                    ),
                }
            )

    expansion_data = pd.DataFrame(expansions).rename(columns=lambda x: f"expansion_{x}")
    avg_exp_len = expansion_data["expansion_length"].mean().round()
    expansion_data = pd.concat(
        [
            expansion_data,
            pd.DataFrame(
                {
                    "expansion_dates": [f"Avg. length: {avg_exp_len} qtrs"],
                    "expansion_length": [avg_exp_len],
                    "expansion_mean_pct_change": [
                        expansion_data["expansion_mean_pct_change"].mean()
                    ],
                }
            ),
        ],
        axis=0,
    )
    contraction_data = pd.DataFrame(contractions).rename(columns=lambda x: f"contraction_{x}")
    avg_cont_len = contraction_data["contraction_length"].mean().round()
    contraction_data = pd.concat(
        [
            contraction_data,
            pd.DataFrame(
                {
                    "contraction_dates": [f"Avg. length: {avg_cont_len} qtrs"],
                    "contraction_length": [avg_cont_len],
                    "contraction_mean_pct_change": [
                        contraction_data["contraction_mean_pct_change"].mean()
                    ],
                }
            ),
        ],
        axis=0,
    )

    chart_df = pd.concat([expansion_data, contraction_data], axis="columns")
    _write_xlsx_sheet(chart_df, "business_cycle_table")

    # Format for HTML
    pd_df = chart_df.drop(  # (pd.DataFrame(table.source.data)
        ["expansion_length", "contraction_length"], axis=1
    )[
        [
            "contraction_dates",
            "contraction_mean_pct_change",
            "expansion_dates",
            "expansion_mean_pct_change",
        ]
    ]
    tbl_html0 = (
        pd_df.style.format(
            {
                "expansion_mean_pct_change": _pd_format_precision("growth_rate_pct"),
                "contraction_mean_pct_change": _pd_format_precision("growth_rate_pct"),
            }
        )
        .relabel_index(["Contractions", "Avg. Change", "Expansions", "Avg. Change"], axis=1)
        .set_table_attributes('class="table table-condensed"')
        .hide(axis="index")
        .to_html()
    )
    tbl_html0 = (
        tbl_html0.replace(' col0"', ' col0" align="left"')
        .replace(' col1"', ' col0" align="center"')
        .replace(' col2"', ' col0" align="left"')
        .replace(' col3"', ' col0" align="center"')
        .replace("%", '<span class="percent">%</span>')
    )
    tbl_html0_lines = tbl_html0.splitlines(keepends=True)
    tbl_html = (
        "".join(tbl_html0_lines[:-8])
        + "</tbody><tfoot>"
        + "".join(tbl_html0_lines[-8:-2])
        + "</tfoot></table>"
    )

    return tbl_html
