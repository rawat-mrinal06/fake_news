import csv
wr = csv.writer(open('train_summ_<s>.csv', 'w'))
for r in csv.reader(open('train_summ.csv')):
    # text = r[0].split('<SEP>')[0]
    # wr.writerow([text.strip(), r[1]])
    wr.writerow([r[0].replace('<SEP>', '</s>'), r[1]])

