def generate_dummy_data():
    """Generate dummy data for fine-tuning."""
    return [
        {
            "instruction": "Analyze this content",
            "input": "Video about cats playing piano got 1M views",
            "output": "This video performed well due to its unique combination of pets and music."
        },
        {
            "instruction": "Suggest content ideas",
            "input": "Current trend: dance challenges",
            "output": "Consider creating a dance challenge featuring unexpected elements like office workers."
        }
        # Add more examples as needed
    ]