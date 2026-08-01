import csv


class CsvLogger:
    def __init__(self, path, columns):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(columns)

    def log(self, *values):
        self.writer.writerow(values)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()
