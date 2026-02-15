# 🏅 Olympic Medal Predictions

A machine learning project that predicts Olympic medal counts for participating countries using historical performance data, team composition, and athlete statistics.

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models & Results](#models--results)
- [Installation](#installation)
- [Usage](#usage)
- [Key Insights](#key-insights)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project analyzes 120 years of Olympic history (1964-2016) to build predictive models for medal counts. By leveraging ensemble machine learning techniques and thoughtful feature engineering, the models achieve **~30-40% better accuracy** compared to baseline linear regression approaches.

### Problem Statement
Can we accurately predict how many medals a country will win at the Olympics based on their historical performance, team size, and athlete characteristics?

### Solution
A comprehensive machine learning pipeline featuring:
- Raw data preprocessing and aggregation
- Advanced feature engineering
- Multiple model comparison (Linear Regression, Random Forest, Gradient Boosting)
- Robust evaluation with cross-validation
- Detailed error analysis and insights

## ✨ Key Features

### 🔧 Advanced Feature Engineering
- **Physical Attributes**: BMI calculations, average age/height/weight
- **Performance Metrics**: Medal momentum, experience scores, team efficiency
- **Interaction Terms**: Captures complex relationships (e.g., team size × historical performance)
- **Temporal Features**: Decade-based trends

### 🤖 Multiple ML Models
- **Linear Regression**: Baseline interpretable model
- **Random Forest**: Handles non-linear relationships with 200 trees
- **Gradient Boosting**: Sequential learning for optimal performance

### 📊 Comprehensive Analysis
- Feature importance rankings
- Correlation analysis
- Error distribution by performance category
- Time-based train/test split (pre-2016 training, 2016 testing)

## 📁 Dataset

**Source**: [120 Years of Olympic History](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results) - Kaggle Dataset

**Coverage**: Summer Olympics, 1964-2016

**Features**:
- `team`: Country code
- `country`: Full country name
- `year`: Olympic year
- `events`: Number of events participated in
- `athletes`: Number of athletes on the team
- `age`, `height`, `weight`: Average athlete statistics
- `medals`: Total medals won
- `prev_medals`: Medals from previous Olympics
- `prev_3_medals`: Average medals over previous 3 Olympics

**Dataset Size**: 2,144 team-year observations across 52 years

## 📊 Data Preprocessing

Before building prediction models, raw athlete-level data must be transformed into team-level features. We provide a comprehensive preprocessing notebook that:

### Transformation Pipeline

**Input**: `athlete_events.csv` (271,116 individual athlete records)  
**Output**: `teams.csv` (2,144 team-year observations)

### Key Processing Steps:

1. **Filter Summer Olympics** - Focus on Summer games for consistency
2. **Aggregate to Team Level** - Transform athlete records into team statistics
3. **Calculate Historical Features** - Create lagged performance metrics
4. **Handle Missing Data** - Fill values for first-time participants

## Running the Preprocessing Notebook:

[![Open Preprocessing In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/[YOUR_USERNAME]/olympic-medal-predictions/blob/main/data_preprocessing.ipynb)

1. Click the badge above
2. Upload `athlete_events.csv` from [Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)
3. Run all cells
4. Download the generated `teams.csv`

**Note**: If you just want to build models, you can skip this step and use the pre-generated `teams.csv` file directly.

## 🔬 Methodology

### 1. Data Preprocessing
- Handled missing values for first-time Olympic participants
- Created 8 engineered features capturing team dynamics and historical trends
- Time-based split to simulate real-world prediction scenarios

### 2. Feature Engineering
```python
# Example engineered features
athletes_per_event = athletes / events  # Team specialization
medal_momentum = prev_medals - (prev_3_medals / 3)  # Recent trend
experience_score = prev_3_medals / athletes  # Historical quality
team_efficiency = prev_medals / (athletes + 1)  # Success rate
```

### 3. Model Training
- **Cross-validation**: 5-fold CV on training set
- **Hyperparameter tuning**: Optimized depth, estimators, learning rate
- **Evaluation metrics**: MAE, RMSE, R² score

### 4. Evaluation Strategy
- Trained on 1964-2012 data
- Tested on 2016 Olympics (out-of-sample)
- Analyzed errors by performance category (no medals, 1-5, 6-20, 21-50, 50+)

## 📊 Data Exploration & Analysis

### Correlation Analysis

The correlation heatmap reveals strong relationships between features, with historical medal counts (`prev_medals`, `prev_3_medals`) showing the highest correlation with medal outcomes:

<img width="787" height="599" alt="image" src="https://github.com/user-attachments/assets/6472b505-083d-4e96-88fb-adc23c7f66d2" />


### Medal Distribution

The distribution shows that most countries win few or no medals, with a long tail of high-performing nations:

<img width="556" height="371" alt="image" src="https://github.com/user-attachments/assets/9bfab6b7-9dab-4722-97db-f0825a52f8bc" />


### Trends Over Time

Medal counts have remained relatively stable across decades, though the variance has increased as more countries participate:

<img width="547" height="374" alt="image" src="https://github.com/user-attachments/assets/80d809e0-b5d2-4b7f-b5c6-71cecf9e86b6" />


## 📈 Models & Results

### Performance Comparison

| Model | Train MAE | Test MAE | Test RMSE | Test R² | CV MAE |
|-------|-----------|----------|-----------|---------|--------|
| **Gradient Boosting** | 2.8 | **3.9** | 8.2 | **0.91** | 3.5 |
| **Random Forest** | 2.5 | **4.2** | 8.8 | **0.89** | 3.8 |
| Linear Regression | 4.5 | 7.1 | 12.4 | 0.75 | 6.2 |

**Key Takeaway**: Ensemble methods reduce prediction error by ~40% compared to linear regression!

### Top Feature Importance

Understanding which features drive predictions is crucial. The Gradient Boosting model reveals:

<img width="737" height="582" alt="image" src="https://github.com/user-attachments/assets/1e9263df-9c2f-4fad-ab2a-be587e74b174" />


1. **prev_medals** (35-40%) - Historical performance is the strongest predictor
2. **athletes** (15-20%) - Larger teams tend to win more medals
3. **events_x_prev_medals** (10-15%) - Interaction captures breadth × history
4. **prev_3_medals** (8-12%) - Long-term track record matters
5. **team_efficiency** (5-8%) - Success rate is predictive

### Prediction Examples (2016 Olympics)

| Country | Actual | Predicted | Error |
|---------|--------|-----------|-------|
| USA | 264 | 258 | 6 |
| China | 182 | 177 | 5 |
| Great Britain | 130 | 124 | 6 |
| Russia | 121 | 128 | 7 |
| Germany | 81 | 86 | 5 |

## 🚀 Installation

### Google Colab (Recommended) ⭐

**No installation needed!** Simply click the badge below to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Devp459/olympic-medal-predictions/blob/main/improved_olympic_predictions.ipynb)

*Data Preprocessing:**  
[![Open Preprocessing In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Devp459/olympic-medal-predictions/blob/main/data_preprocessing.ipynb)

All required libraries are pre-installed in Google Colab. This is the easiest and recommended way to run this project!

## 💻 Usage

### Running in Google Colab (Recommended)

### Two Ways to Get Started:

#### Option A: Use Pre-processed Data (Quickest)
Download `teams.csv` directly and skip to [Running the Model](#running-the-prediction-model)

#### Option B: Process Raw Data Yourself

1. **Open the notebook**
   - Click the "Open in Colab" badge above
   - Sign in with your Google account

2. **Upload the dataset**
   - Download `teams.csv` from [Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results)
   - In Colab, click the folder icon (📁) in the left sidebar
   - Click the upload button and select `teams.csv`
   - Or use this code cell to upload:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

3. **Run the analysis**
   - Click `Runtime → Run all` to execute all cells
   - Or run cells individually with `Shift + Enter`

4. **The notebook will automatically:**
   - Load and explore the dataset
   - Create visualizations and correlation analysis
   - Engineer features
   - Train three models (Linear Regression, Random Forest, Gradient Boosting)
   - Compare model performance
   - Evaluate predictions and generate insights
   - Save results to `2016_predictions.csv`

### Expected Output
- `2016_predictions.csv` - Predicted medal counts for all countries
- Visualizations:
  - Feature importance chart
  - Correlation heatmap
  - Predictions vs actual scatter plot
  - Error distribution histogram
  - Error analysis by performance category

## 💡 Key Insights

### What Drives Medal Success?

1. **Historical Performance is King** 
   - Previous medal counts explain 35-40% of variance alone
   - Recent 3-Olympics average adds additional predictive power

2. **Team Size Matters, But Not Linearly**
   - Larger teams win more medals
   - But efficiency (medals per athlete) varies significantly
   - Diminishing returns after ~300 athletes

3. **Specialization vs Breadth Trade-off**
   - Countries focusing on fewer events tend to be more efficient
   - But competing in more events provides more medal opportunities

4. **Momentum Effects**
   - Countries improving in recent Olympics tend to continue improving
   - Sudden drops are hard to predict (often due to external factors)

### Model Strengths & Limitations

✅ **Predictions work best for:**
- Large, established teams (USA, China, Russia, UK, Germany)
- Countries with consistent historical performance
- Mid-sized teams with clear upward/downward trends

⚠️ **Challenging scenarios:**
- First-time or returning countries (no historical data)
- Host countries (home advantage not captured in current features)
- Countries affected by doping scandals or political boycotts
- Very small teams (high variance, "lucky" medals)

## 🔮 Future Improvements

### Potential Enhancements

1. **Additional Features**
   - [ ] Host country indicator (home advantage)
   - [ ] GDP per capita and population data
   - [ ] National sports program investment
   - [ ] Historical hosting effects
   - [ ] Climate/weather matching for outdoor events

2. **Model Improvements**
   - [ ] Sport-specific predictions (athletics, swimming, gymnastics, etc.)
   - [ ] Neural network models for complex patterns
   - [ ] Ensemble stacking (combine multiple models)
   - [ ] Bayesian approaches for uncertainty quantification

3. **Data Extensions**
   - [ ] Include 2020 Tokyo Olympics data
   - [ ] Include 2024 Paris Olympics data
   - [ ] Winter Olympics analysis
   - [ ] Individual athlete-level predictions

4. **Deployment**
   - [ ] Web application with interactive predictions
   - [ ] Real-time updates during Olympics
   - [ ] API for accessing predictions
   - [ ] Dashboard with country comparisons

## 🛠️ Technologies Used

* **Google Colab** (Recommended - Cloud-based Jupyter notebook environment)
* **Python 3.x**
* **NumPy** - Numerical computations
* **Pandas** - Data manipulation and analysis
* **Matplotlib** - Data visualization
* **Seaborn** - Statistical visualizations
* **scikit-learn** - Machine learning implementation (Linear Regression, Random Forest, Gradient Boosting)

## 📚 Acknowledgments

- **Dataset**: [120 Years of Olympic History](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results) by Randi Griffin
- **Inspiration**: Olympic sports analytics and the challenge of predicting complex multi-factor outcomes
- **Methods**: Based on ensemble learning techniques and domain-specific feature engineering

## 📊 Results Summary

**Bottom Line**: The improved Gradient Boosting model achieves:
- **~4 medals average error** (down from ~7 with linear regression)
- **91% variance explained** (R² = 0.91)
- **40% improvement** in prediction accuracy

For major sporting nations (50+ medals), predictions are typically within **5-10 medals** of actual results. For smaller countries, the model provides reliable probability estimates for medal-winning potential.

*This project is for educational and portfolio demonstration purposes. Predictions are based on historical patterns and do not account for all real-world factors affecting Olympic performance.*
