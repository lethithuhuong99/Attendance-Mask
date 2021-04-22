import pymongo
import csv

myclient = pymongo.MongoClient("mongodb+srv://admin:admin123@cluster0.9n8yb.mongodb.net/employees?retryWrites=true&w=majority")
mydb = myclient['employees']
mycol = mydb['attendancedbs']

with open('Attendance/Attendance_2021-04-20_17-25-57.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    attendanceList = []

    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            attendanceDetail = {
                "date": row[0],
                "userId": row[1],
                "mask": row[2],
                "checkIn": row[3],
                "checkOut": row[4],
            }
            attendanceList.append(attendanceDetail)
            line_count += 1
    print(f'Processed {line_count} lines.')

x = mycol.insert_many(attendanceList)
