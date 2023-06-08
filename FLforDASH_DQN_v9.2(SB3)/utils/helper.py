from csv import writer
import matplotlib.pyplot as plt


# def write2csv(file, data):
#     with open('{}.csv'.format(file), mode='w', newline='') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerows(data)

def list2csv(file, datalist):
    with open('{}.csv'.format(file), mode='a', newline='') as csv_file:
        writer_obj = writer(csv_file)
        writer_obj.writerow(datalist)

def list2newcsv(file, datalist):
    with open('{}.csv'.format(file), mode='w+') as csv_file:
        writer_obj = writer(csv_file)
        writer_obj.writerow(datalist)
    csv_file.close()

def list2newcsv_col(file, datalist):
    with open('{}.csv'.format(file), mode='w+', newline='') as csv_file:
        writer_obj = writer(csv_file)
        for val in datalist:
            writer_obj.writerow([val])
    csv_file.close()

def twodlist2csv(file, twodlist2csv):
    with open('{}.csv'.format(file), mode='a', newline='') as csv_file:
        writer_obj = writer(csv_file)
        for r in twodlist2csv:
            writer_obj.writerow(r)

def plot_reward(file, reward_trace):
    plt.figure(figsize=(15, 3))
    plt.plot(reward_trace)
    plt.savefig(file)
    plt.clf()


if __name__ == '__main__':
    # list2csv('text', ['John Smith', 'Accounting', 'November'])
    # list2csv('text', ['John Smith1', 'Accounting1', 'November1'])

    a = [[1, 2], [3, 4]]
    twodlist2csv('text', a)
