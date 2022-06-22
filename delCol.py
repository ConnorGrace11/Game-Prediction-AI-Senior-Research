import csv

with open('addedGames.csv',"r") as fin:
    with open('delGames.csv',"w") as fout:
        writer=csv.writer(fout)
        for row in csv.reader(fin):
            writer.writerow(row[:-1])
