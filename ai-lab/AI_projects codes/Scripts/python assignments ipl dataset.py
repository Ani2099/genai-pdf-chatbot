import pandas as pd
ipl = pd.read_csv("ipl.csv")

Q1> Frquency of Player_of_Match of every season

ipl['Player_of_Match'].value_counts()

Q2> Highest man of the match award :

mom_counts = ipl['Player_of_Match'].value_counts()
highest_mom_player = mom_counts.idxmax()
highest_mom_count = mom_counts.max()
display(f"Player with the highest 'Player of the Match' awards: {highest_mom_player} ({highest_mom_count} awards)")

Q3> Highest no of matches:

team_matches = pd.concat([ipl['Team1'], ipl['Team2']]).value_counts()
highest_matches_team = team_matches.idxmax()
highest_matches_count = team_matches.max()
display(f"Team with the highest number of matches played: {highest_matches_team} ({highest_matches_count} matches)")

Q4> How many times CSK won by runs and what is the highest margin?

csk_wins_by_runs = ipl[(ipl['WinningTeam'] == 'Chennai Super Kings') & (ipl['WonBy'] == 'Runs')]
num_csk_wins_by_runs = len(csk_wins_by_runs)
highest_margin_csk_runs = csk_wins_by_runs['Margin'].max()

display(f"Chennai Super Kings won by runs {num_csk_wins_by_runs} times.")
display(f"Their highest winning margin by runs is {highest_margin_csk_runs}.")

Q5> Team won by highest no of wickets:

won_by_wickets = ipl[ipl['WonBy'] == 'Wickets']
highest_wickets_win = won_by_wickets.loc[won_by_wickets['Margin'].idxmax()]
winning_team = highest_wickets_win['WinningTeam']
margin = highest_wickets_win['Margin']
display(f"Team that won by the highest number of wickets: {winning_team} (by {margin} wickets)")

Q6> how many have hosted the matches?

num_cities = ipl['City'].nunique()
display(f"The matches were hosted in {num_cities} different cities.")

Q7> How many times mumbai loose toss?


mumbai_matches = ipl[(ipl['Team1'] == 'Mumbai Indians') | (ipl['Team2'] == 'Mumbai Indians')]
mumbai_lost_toss = mumbai_matches[mumbai_matches['TossWinner'] != 'Mumbai Indians']
num_mumbai_lost_toss = len(mumbai_lost_toss)
display(f"Mumbai Indians lost the toss {num_mumbai_lost_toss} times.")

Q8> Which team won the trophy and by which margin?


trophy_match = ipl.iloc[-1]
winning_team = trophy_match['WinningTeam']
won_by = trophy_match['WonBy']
margin = trophy_match['Margin']

display(f"The team that won the trophy is {winning_team}, winning by {margin} {won_by}.")