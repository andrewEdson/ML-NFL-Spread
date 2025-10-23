import nfl_data_py as nfl
import pandas as pd

df = nfl.import_schedules([2023])

print(df.columns)

team_stats = nfl.import_weekly_data([2023])
print(team_stats.columns)
