import xml.etree.ElementTree as ET
import numpy as np
import collections


class SubmissionParser:
    def __init__(self, xml_file_path, csv_directory, n_rows, delim):
        self.xml_file_path = xml_file_path
        self.csv_directory = csv_directory
        self.n_rows = n_rows
        self.delim = delim

        self.n_cols = 3

    def parse_submission(self, elem):
        # This function gets an element node from ElementTree
        # representing a single submission and parses it into one csv row

        # Put all the children of elem into a queue
        # It is important we only iterate over the direct children, because
        # submissions can be nested
        q = collections.deque(elem)
        row = [None] * self.n_cols

        while q:
            node = q.popleft()

            if node.tag == 'ReviewStatus':
                row[0] = node.text
                # Classification is always next. Can't check for tag because it
                # can have multiple tag names
                classification = q.popleft()
                row[1] = classification.text

            if node.tag == 'Comment':
                row[2] = node.text

            if None not in row:
                return row

        # Incomplete entry (perhaps simple allele)
        return None

    def parse(self):
        counter = 0
        dataset = [['ReviewStatus', 'Classification', 'Comment']]
        # We parse the tree at the variant level
        for event, elem in ET.iterparse(self.xml_file_path):
            # Only get the records which have an associated comment
            # This XML path gets the nodes which have a child 'Comment'
            submissions = elem.findall(".//Comment/..")
            for submission in submissions:
                row = self.parse_submission(submission)
                if row is not None:
                    dataset.append(row)
                counter += 1
            if counter >= self.n_rows:
                break

        return np.array(dataset)

    def save_dataset(self, dataset):
        np.save(self.csv_directory + 'submissions.npy', dataset)


# parser = SubmissionParser(
#    "ClinVarVCVRelease_2024-02.xml", "csv/", 10000, '~')
# parser.parse()
