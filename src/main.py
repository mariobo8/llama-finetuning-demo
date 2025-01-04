from data.dummy_data import generate_dummy_data
from models.trainer import LlamaFineTuner

def main():
    # Generate dummy data
    data = generate_dummy_data()
    
    # Initialize fine-tuner
    fine_tuner = LlamaFineTuner()
    fine_tuner.setup()
    
    # Prepare and train
    dataset = fine_tuner.prepare_data(data)
    fine_tuner.train(dataset)

if __name__ == "__main__":
    main()