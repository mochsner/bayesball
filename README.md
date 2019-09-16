# BAYESBALL
Attempting to predict a Win/Loss for any given baseball game.

Making use of any classification data: discrete or categorical.

Tools:
- genie modeler
- python: numpy, pandas, sklearn, seaborn
  - NOT pgmpy (maximum_likelihood_estimator.getparams err)

## Considerations:
- Park factor (Home or Away)
- Opposing team (as pertaining to historical data... but how?)
- How to take into account the redundancy of home/away affecting that data? Can't assume independence for Naive Bayes
- Maybe only consider data with matching parameters team A, team B, field A/B (NOTE: TEAM_ID == PARK_ID)
