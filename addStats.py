

#Trying to add in data automatically so that it is less time consuming for any future data
myData = {
         'winLoss': 'Defense',
        'Houston Texans': (72, 78),
        'Jacksonville Jaguars': (75, 76),
        'Kansas City Chiefs': (78, 96),
        'Miami Dolphins': (91, 77),
        'Denver Broncos': (86, 72),
        'Green Bay Packers': (85, 91),
        'Dallas Cowboys': (76, 88),
        'Pittsburgh Steelers': (88, 74),
        'Washington': (81, 76),
        'New York Giants': (82, 72),
        'Philadelphia Eagles': (72, 78),
        'Tennessee Titans': (80, 87),
        'Indianapolis Colts': (78, 75),
        'Las Vegas Raiders': (75, 82),
        'Las Angeles Chargers': (80, 82),
        'Buffalo Bills': (82, 86),
        'New England Patriots': (86, 78),
        'New York Jets': (74, 69),
        'Cincinnati Bengals': (74, 74),
        'Cleveland Browns': (79, 89),
        'Baltimore Ravens': (85, 84),
        'Minnesota Vikings': (81, 77),
        'Chicago Bears': (87, 73),
        'Detroit Lions': (74, 75),
        'New Orleans Saints': (83, 80),
        'Carolina Panthers': (76, 71),
        'Atlanta Falcons': (73, 79),
        'Los Angeles Rams': (83, 81),
        'Seattle Seahawks': (77, 85),
        'Arizona Cardinals': (84, 83),
        'San Francisco 49ers': (84, 80),
        'Tampa Bay Buccaneers': (90, 92),
}
import csv
with open('games.csv','r') as csvinput:
    with open('addedGames.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        for row in csv.reader(csvinput):
            writer.writerow(row+['Berry'])
#df = pd.read_csv('games.csv')
#df
