#import pandas as pd

myData = {
         'winLoss': 'Defense',
        'Houston Texans': 72,
        'Jacksonville Jaguars': 75,
        'Kansas City Chiefs': 78,
        'Miami Dolphins': 91,
        'Denver Broncos': 86,
        'Green Bay Packers': 85,
        'Dallas Cowboys': 76,
        'Pittsburgh Steelers': 88,
        'Washington': 81,
        'New York Giants': 82,
        'Philadelphia Eagles': 72,
        'Tennessee Titans': 80,
        'Indianapolis Colts': 78,
        'Las Vegas Raiders': 75,
        'Los Angeles Chargers': 80,
        'Buffalo Bills': 82,
        'New England Patriots': 86,
        'New York Jets': 74,
        'Cincinnati Bengals': 74,
        'Cleveland Browns': 79,
        'Baltimore Ravens': 85,
        'Minnesota Vikings': 81,
        'Chicago Bears': 87,
        'Detroit Lions': 74,
        'New Orleans Saints': 83,
        'Carolina Panthers': 76,
        'Atlanta Falcons': 73,
        'Los Angeles Rams': 83,
        'Seattle Seahawks': 77,
        'Arizona Cardinals': 84,
        'San Francisco 49ers': 84,
        'Tampa Bay Buccaneers': 90,

}

oData = {
         'winLoss': 'Defense',
        'Houston Texans': 78,
        'Jacksonville Jaguars': 76,
        'Kansas City Chiefs': 96,
        'Miami Dolphins': 77,
        'Denver Broncos': 72,
        'Green Bay Packers': 91,
        'Dallas Cowboys': 88,
        'Pittsburgh Steelers': 74,
        'Washington': 76,
        'New York Giants': 72,
        'Philadelphia Eagles': 78,
        'Tennessee Titans': 87,
        'Indianapolis Colts': 75,
        'Las Vegas Raiders': 82,
        'Los Angeles Chargers': 82,
        'Buffalo Bills': 86,
        'New England Patriots': 78,
        'New York Jets': 69,
        'Cincinnati Bengals': 74,
        'Cleveland Browns': 89,
        'Baltimore Ravens': 84,
        'Minnesota Vikings': 77,
        'Chicago Bears': 73,
        'Detroit Lions': 75,
        'New Orleans Saints': 80,
        'Carolina Panthers': 71,
        'Atlanta Falcons': 79,
        'Los Angeles Rams': 81,
        'Seattle Seahawks': 85,
        'Arizona Cardinals': 83,
        'San Francisco 49ers': 80,
        'Tampa Bay Buccaneers': 92,

}

import csv
with open('games.csv','r') as csvinput:
    with open('total.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        for row in csv.reader(csvinput):
            if row[1] in myData.keys() and row[4] in oData.keys():
                writer.writerow(row + [myData[row[1]]] + [myData[row[4]]] + [oData[row[1]]] +[oData[row[4]]])
            #writer.writerow(row+['Berry'])
#df = pd.read_csv('games.csv')
#df
