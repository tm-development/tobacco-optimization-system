# Tobacco Optimization System

This project is a web application designed for optimizing tobacco production parameters using various algorithms. It provides a user-friendly interface for data import, model management, parameter optimization, and result analysis.

## Project Structure

```
tobacco-optimization-system
├── app
│   ├── __init__.py        # Initializes the Flask application
│   ├── routes.py          # Contains the route definitions
│   ├── models.py          # Defines the data models
│   ├── forms.py           # Contains form definitions for user input
│   └── utils.py           # Utility functions for the application
├── templates
│   └── index.html         # Main HTML template for the application
├── static
│   ├── css
│   │   └── style.css      # CSS styles for the application
│   ├── js
│   │   └── main.js        # JavaScript for interactive features
│   └── images             # Directory for image assets
├── config.py              # Configuration settings for the application
├── requirements.txt       # List of required Python packages
├── run.py                 # Entry point to run the application
└── README.md              # Documentation and usage instructions
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd tobacco-optimization-system
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the application, execute the following command:
```
python run.py
```

Visit `http://127.0.0.1:5000` in your web browser to access the application.

## Features

- Data import from CSV and Excel files
- Model management for predictive analytics
- Parameter optimization using algorithms like Particle Swarm Optimization
- Result analysis and visualization

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.