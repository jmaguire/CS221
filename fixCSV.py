import csv
import numpy as np
doc = []
results = {}
with open('Data/BLEU_Score.csv', 'r') as f:
    reader = csv.reader(f,delimiter = ',')
    reader.next()
    for row in reader:
        key = row[0] 
        values = np.array([float(elem) for elem in row[1:]])
        # print values
        if key not in results:
            results[key] = values
        else:
            results[key] += values
title = ['Filename','BLEU Score LDA', 'BLEU Score Frequency', 'Calibration']
with open('Data/BLEU_ScoreFIXed.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(title)
        for key in results:
            writer.writerow([key]+list(results[key]/10.))