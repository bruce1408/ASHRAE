import xlrd
import csv
import pandas as pd


def csv_from_excel():
    wb = xlrd.open_workbook('/home/bruce/PycharmProjects/ASHRAE/data/traffic.xlsx')
    sh = wb.sheet_by_name('出站记录明细')
    your_csv_file = open('traffic_v1.csv', 'w')
    wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

    for rownum in range(sh.nrows):
        wr.writerow(sh.row_values(rownum))

    your_csv_file.close()

csv_from_excel()