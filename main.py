from MachinePricePredictor import MachinePricePredictor
import argparse

if __name__ == "__main__":
    # Define the parser
    parser = argparse.ArgumentParser(description='Machine Price Predictor')
    parser.add_argument('--data_path', type=str, default='data/machines.csv', help='Path to the file containing the data. Default is data/machines.csv')
    parser.add_argument('--visualize_preprocessing',
                        action='store_true',
                        help='Visualize data before preprocessing. It will create a visualization directory in the current directory with the plots')
    parser.add_argument('--visualize_postprocessing',
                        action='store_true',
                        help='Visualize data after preprocessing. It will create a visualization directory in the current directory with the plots')
    args = parser.parse_args()

    DATA_PATH = args.data_path
    POSTPROCESSING = args.visualize_postprocessing

    # initialize the predictor
    predictor = MachinePricePredictor(data_path= DATA_PATH)

    # visualize
    if args.visualize_preprocessing:
        print('Preprocessing steps visualized...')
        # if the user wants to visualize the data before preprocessing
        predictor.visualize(prefeature=True)
    if args.visualize_postprocessing:
        print('Postprocessing steps visualized...')
        # if the user wants to visualize the data after preprocessing
        predictor.visualize(prefeature=False)

    # train the model and evaluate it
    y_test, y_pred = predictor.train()

    # evaluate the model
    predictor.evaluate_model(y_test, y_pred)