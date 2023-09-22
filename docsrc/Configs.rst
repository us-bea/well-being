Configuration files
===================

This package was designed to be easily editted by non-programmers by storing many configuration details in human-editable (YAML) configuration files.

1. ``wellbeing/bokeh_theme.yml`` - These stores the majority of the theming details for charts. For more information on what can be specified here, see https://docs.bokeh.org/en/latest/docs/reference/themes.html#theme. Some BEA-style colors are hard-coded in ``generate_chart.py``.
2. ``wellbeing/chart_config.yml`` - There is some common configuration details at the top (default chart size, background color, date for narratives). Below that are chart-specific sections that list chart/table metadata (title, subtitle, notes, narrative), default parameters (e.g., ``start_year``, ``end_year``), and then details of API requests used to collect data (less likely to be modified).
