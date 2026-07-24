from dataset.dataset_generator import DatasetGenerator


def main():

    dataset = DatasetGenerator(

        output_directory="datasets",

        number_of_samples=5,

        cube_size=(64, 64, 64),

        random_seed=42

    )

    dataset.generate_dataset()


if __name__ == "__main__":

    main()