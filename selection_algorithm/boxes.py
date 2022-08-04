
from math import sqrt
import math
import numpy as np
import pandas as pd


class Boxes:

    def __init__(self, dimension, divisions, dataset_num_limit, distance_threshold_for_train):
        self.dimension = dimension
        self.divisions = divisions
        self.box_size = 1 / (divisions + 1)
        self.num_of_samples_in_boxes = 0
        self.dataset_num_limit = dataset_num_limit
        self.distance_threshold_for_train = distance_threshold_for_train
        self.dataset_columns = []
        self.current_center_of_mass = None
        self.dataset_changes_threshold = int(self.dataset_num_limit / 0.3)
        self.dataset_changes = 0
        self.boxes = []
        self.boxes_coordinates = {}
        self.boxes_index = {}
        for _ in range((divisions + 1) ** dimension):
            self.boxes.append([])

    def set_init_dataset(self, dataset):
        self.dataset_columns = dataset['x'].columns
        for i in range(len(dataset['x'])):
            self.handle({
                'x': dataset['x'].iloc[i].values,
                'y': dataset['y'].iloc[i]
            }, 0)
        self.current_center_of_mass = self.calc_center_of_mass()

    def calc_box_index(self, coordinates):
        index = 0
        qtd_box_in_one_side = self.divisions + 1
        coordinates_len = len(coordinates)
        for i in range(coordinates_len):
            index += coordinates[i] * (qtd_box_in_one_side ** i)
        return index

    def calc_box_coordinate(self, box_index):
        if not box_index in self.boxes_coordinates:
            qtd_box_in_one_side = self.divisions + 1
            coordinates = []
            index = box_index
            for i in reversed(range(self.dimension)):
                value = int(index / (qtd_box_in_one_side ** i))
                index = index % (qtd_box_in_one_side ** i)
                coordinates.append(value)
            self.boxes_coordinates[box_index] = list(reversed(coordinates))

        return self.boxes_coordinates[box_index]

    def calc_sample_coordinates(self, sample):
        coordinates = []
        for current_value in sample:
            for index in range(self.divisions + 1):
                bottom_limit = index * self.box_size
                top_limit = (index + 1) * self.box_size
                if bottom_limit <= current_value and current_value <= top_limit:
                    coordinates.append(index)
                    break
        return coordinates

    def get_fuller_box_index(self):
        current_index = 0
        current_size = 0
        for index, box in enumerate(self.boxes):
            box_size = len(box)
            if box_size > current_size:
                current_size = box_size
                current_index = index
        return current_index

    def get_dataset(self):
        filtered_box = list(filter(lambda box: len(box) > 0, self.boxes))
        flat_dataset = [x for xs in filtered_box for x in xs]
        x = list(map(lambda sample: sample['x'], flat_dataset))
        y = list(map(lambda sample: sample['y'], flat_dataset))
        return {
            'x': pd.DataFrame(x, columns=self.dataset_columns),
            'y': pd.DataFrame(data=y)
        }

    def calc_center_of_mass(self):
        center = [0] * self.dimension
        for index in range(self.dimension):
            mass = 0
            mass_x_distance = 0
            for box_index, box in enumerate(self.boxes):
                current_mass = len(box) / self.dataset_num_limit
                mass += current_mass
                mass_x_distance += current_mass * \
                    self.calc_box_coordinate(box_index)[index]
            center[index] = mass_x_distance / mass
        return center

    def calc_distance(self, new_center):
        distance = 0
        for index, new_center_coordinate in enumerate(new_center):
            distance += (self.current_center_of_mass[index] -
                         new_center_coordinate) ** 2
        return sqrt(distance)

    def should_retrain(self):
        if self.dataset_changes_threshold < self.dataset_changes:
            self.dataset_changes = 0
            new_center = self.calc_center_of_mass()
            distance = self.calc_distance(new_center)
            if distance > self.distance_threshold_for_train:
                self.current_center_of_mass = new_center
                return True
        return False

    def reduce_dimension(self, sample):
        chunk_size = math.ceil(len(sample) / self.dimension)
        chunks = [sample[i:i+chunk_size]
                  for i in range(0, len(sample), chunk_size)]

        return list(map(lambda chunk: np.array(chunk).mean(), chunks))

    def handle(self, sample, index):

        self.dataset_changes += 1

        sample['x_reduced'] = self.reduce_dimension(sample['x'])
        coordinates = self.calc_sample_coordinates(sample['x_reduced'])
        box_index = self.calc_box_index(coordinates)

        self.boxes[box_index].append(sample)
        self.num_of_samples_in_boxes += 1

        if self.num_of_samples_in_boxes > self.dataset_num_limit:
            fuller_box_index = self.get_fuller_box_index()
            self.boxes[fuller_box_index].pop(0)
            self.num_of_samples_in_boxes -= 1
