import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)

OUTPUT_DIR = r'C:\Users\HP\Searches\Internship\capstone_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
PDF_PATH = os.path.join(OUTPUT_DIR, 'Final_Capstone_Project_Telco_Churn.pdf')


def ensure_directory():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_clean_data():
    path = r'C:\Users\HP\Searches\Internship\P_4_WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].str.strip()
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['Churn_Code'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df


def save_churn_distribution_plot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Churn', palette=['#2ecc71', '#e74c3c'], ax=ax)
    ax.set_title('Customer Churn Distribution')
    ax.set_xlabel('Churn status')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'churn_distribution.png')
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def save_contract_analysis_plot(df):
    contract_churn = pd.crosstab(df['Contract'], df['Churn'])
    fig, ax = plt.subplots(figsize=(8, 5))
    contract_churn.plot(kind='bar', stacked=True, ax=ax, color=['#2ecc71', '#e74c3c'])
    ax.set_title('Churn by Contract Type')
    ax.set_xlabel('Contract')
    ax.set_ylabel('Count')
    ax.legend(['No', 'Yes'])
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'contract_churn.png')
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def save_charges_boxplot(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x='Churn', y='MonthlyCharges', palette=['#2ecc71', '#e74c3c'], ax=ax)
    ax.set_title('Monthly Charges by Churn Status')
    ax.set_xlabel('Churn')
    ax.set_ylabel('Monthly Charges ($)')
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, 'monthly_charges_churn.png')
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def calculate_summary_statistics(df):
    churn_rate = round(df['Churn_Code'].mean() * 100, 2)
    avg_tenure = round(df['tenure'].mean(), 2)
    avg_monthly = round(df['MonthlyCharges'].mean(), 2)
    avg_total = round(df['TotalCharges'].mean(), 2)
    churn_by_contract = df.groupby('Contract')['Churn_Code'].mean().sort_values(ascending=False)
    contract_summary = churn_by_contract.round(3).to_dict()

    summary = {
        'Dataset rows': len(df),
        'Dataset columns': len(df.columns),
        'Average tenure (months)': avg_tenure,
        'Average monthly charges ($)': avg_monthly,
        'Average total charges ($)': avg_total,
        'Churn rate (%)': churn_rate,
        'Highest churn contract': next(iter(contract_summary.items()))[0],
        'Highest churn rate': round(next(iter(contract_summary.items()))[1] * 100, 2),
    }
    return summary


def build_model(df):
    X = df.drop(columns=['customerID', 'Churn', 'Churn_Code'])
    y = df['Churn_Code']

    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        (
            'num',
            Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
            ]),
            num_cols,
        ),
        (
            'cat',
            Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
            ]),
            cat_cols,
        ),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        'Accuracy': round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall': round(recall_score(y_test, y_pred), 4),
        'F1 Score': round(f1_score(y_test, y_pred), 4),
        'ROC AUC': round(roc_auc_score(y_test, y_prob), 4),
    }

    labels = ['No', 'Yes']
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion = {
        'True Negative': int(tn),
        'False Positive': int(fp),
        'False Negative': int(fn),
        'True Positive': int(tp),
    }

    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    coefficients = pipeline.named_steps['model'].coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
    })
    feature_importance['Abs_Coefficient'] = feature_importance['Coefficient'].abs()
    top_features = feature_importance.sort_values('Abs_Coefficient', ascending=False).head(10)

    classification = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    return pipeline, metrics, confusion, top_features, classification


def chi_square_analysis(df):
    contingency = pd.crosstab(df['Contract'], df['Churn'])
    chi2, p_value, _, _ = chi2_contingency(contingency)
    return contingency, chi2, p_value


def save_metrics_csv(metrics, confusion, classification):
    metrics_df = pd.DataFrame([
        ['Accuracy', metrics['Accuracy']],
        ['Precision', metrics['Precision']],
        ['Recall', metrics['Recall']],
        ['F1 Score', metrics['F1 Score']],
        ['ROC AUC', metrics['ROC AUC']],
    ], columns=['Metric', 'Value'])
    metrics_csv = os.path.join(OUTPUT_DIR, 'model_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)

    conf_df = pd.DataFrame([
        ['True Negative', confusion['True Negative']],
        ['False Positive', confusion['False Positive']],
        ['False Negative', confusion['False Negative']],
        ['True Positive', confusion['True Positive']],
    ], columns=['Category', 'Count'])
    conf_csv = os.path.join(OUTPUT_DIR, 'confusion_matrix.csv')
    conf_df.to_csv(conf_csv, index=False)

    class_df = pd.DataFrame(classification).T.reset_index().rename(columns={'index': 'Class'})
    class_csv = os.path.join(OUTPUT_DIR, 'classification_report.csv')
    class_df.to_csv(class_csv, index=False)


def generate_pdf_report(df, summary, metrics, confusion, top_features, contingency, chi2, p_value,
                       churn_plot, contract_plot, charges_plot):
    doc = SimpleDocTemplate(
        PDF_PATH,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=20, leading=22, alignment=1)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#2F4F4F'))
    body_style = styles['BodyText']
    story = []

    story.append(Paragraph('Final Capstone Project', title_style))
    story.append(Paragraph('Customer Churn Prediction in a Telecom Company', styles['Title']))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph('Prepared by: AI Data Science Capstone Project', body_style))
    story.append(Paragraph('Dataset: Telco Customer Churn Dataset', body_style))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph('Executive Summary', heading_style))
    story.append(Paragraph(
        'This project investigates the drivers of customer churn in a telecommunications company using the Telco Customer Churn dataset. '
        'The objective is to predict churn behavior, uncover the most influential customer attributes, and recommend retention actions based on a data-driven analysis.',
        body_style,
    ))
    story.append(PageBreak())

    story.append(Paragraph('1. Problem Definition', heading_style))
    story.append(Paragraph(
        'Customer churn is a major risk for telecom operators because it reduces recurring revenue and increases customer acquisition costs. '
        'The business objective is to identify the customers most likely to leave so that the company can target them with timely interventions and retention programs.',
        body_style,
    ))

    story.append(Paragraph('2. Dataset Understanding', heading_style))
    dataset_table = Table([
        ['Attribute', 'Value'],
        ['Rows', str(summary['Dataset rows'])],
        ['Columns', str(summary['Dataset columns'])],
        ['Target', 'Churn'],
        ['Churn rate', f"{summary['Churn rate (%)']}%"],
        ['Average tenure', f"{summary['Average tenure (months)']} months"],
        ['Average monthly charges', f"${summary['Average monthly charges ($)']:.2f}"],
    ], colWidths=[2.8 * inch, 3.0 * inch])
    dataset_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D9EAF7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(dataset_table)
    story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph('3. Data Cleaning and Preprocessing', heading_style))
    story.append(Paragraph(
        'The raw dataset contained a string-based TotalCharges field that needed conversion to numeric type before analysis. Missing values were treated with a median imputation strategy, '
        'while categorical variables were encoded using one-hot encoding and numerical features were standardized for modeling. The customerID field was removed because it was unique and not predictive of churn.',
        body_style,
    ))

    story.append(Paragraph('4. Exploratory Data Analysis (EDA)', heading_style))
    story.append(Image(churn_plot, width=5.5 * inch, height=3.2 * inch))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Image(contract_plot, width=6.0 * inch, height=3.5 * inch))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Image(charges_plot, width=5.5 * inch, height=3.2 * inch))
    story.append(Paragraph(
        'The target class is imbalanced, with churn accounting for about 26.5% of customers. Analysis shows that customers on month-to-month contracts have a substantially higher churn rate than those with annual contracts. '
        'Additionally, monthly charges are noticeably higher among customers who leave, suggesting that pricing pressure may contribute to churn risk.',
        body_style,
    ))

    story.append(Paragraph('5. Statistical Analysis', heading_style))
    story.append(Paragraph(
        f"A chi-square test of association between Contract and Churn produced chi-square = {chi2:.4f} with p-value = {p_value:.4e}. Because the p-value is well below 0.05, the relationship between contract type and churn is statistically significant. This supports the interpretation that contract structure is an important driver of attrition.",
        body_style,
    ))
    contingency_table = Table([
        ['Contract / Churn', 'No', 'Yes'],
        ['Month-to-month', str(contingency.loc['Month-to-month', 'No']), str(contingency.loc['Month-to-month', 'Yes'])],
        ['One year', str(contingency.loc['One year', 'No']), str(contingency.loc['One year', 'Yes'])],
        ['Two year', str(contingency.loc['Two year', 'No']), str(contingency.loc['Two year', 'Yes'])],
    ], colWidths=[1.8 * inch, 1.3 * inch, 1.3 * inch])
    contingency_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6F4EA')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(contingency_table)

    story.append(Paragraph('6. Machine Learning Model Development', heading_style))
    story.append(Paragraph(
        'A logistic regression classifier was trained to predict churn using a pipeline that combined preprocessing for numerical and categorical variables. '
        'The model was trained on 80% of the data and evaluated on the remaining 20% to avoid overfitting and maintain realistic business performance estimates.',
        body_style,
    ))

    story.append(Paragraph('7. Model Evaluation', heading_style))
    metrics_table = Table([
        ['Metric', 'Value'],
        ['Accuracy', f"{metrics['Accuracy']:.4f}"],
        ['Precision', f"{metrics['Precision']:.4f}"],
        ['Recall', f"{metrics['Recall']:.4f}"],
        ['F1 Score', f"{metrics['F1 Score']:.4f}"],
        ['ROC AUC', f"{metrics['ROC AUC']:.4f}"],
    ], colWidths=[2.4 * inch, 2.0 * inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8F1D4')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(
        f"The confusion matrix indicates {confusion['True Positive']} churners correctly identified, with {confusion['False Negative']} missed churners and {confusion['False Positive']} non-churners incorrectly flagged. The model demonstrates strong predictive power for customer retention planning, especially when used with business rules for proactive outreach.",
        body_style,
    ))

    story.append(Paragraph('8. Key Drivers of Churn', heading_style))
    top_rows = [['Feature', 'Coefficient']] + [
        [str(row['Feature']), f"{row['Coefficient']:.4f}"]
        for _, row in top_features.head(5).iterrows()
    ]
    top_table = Table(top_rows, colWidths=[3.4 * inch, 1.5 * inch])
    top_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F3E5F5')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
    ]))
    story.append(top_table)

    story.append(Paragraph('9. Findings and Recommendations', heading_style))
    story.append(Paragraph(
        '1. Customers on month-to-month contracts are much more likely to churn. The company should focus retention offers on this segment first.\n'
        '2. High monthly charges are associated with churn risk, suggesting the need for pricing review or loyalty discounts.\n'
        '3. Customers with older contracts and stable service bundles are less likely to leave; retention strategies should prioritize vulnerable segments before they cancel.\n'
        '4. Predictive churn scoring can support targeted retention campaigns that offer contract upgrades, service bundles, or loyalty benefits to at-risk customers.',
        body_style,
    ))

    story.append(Paragraph('10. Conclusion', heading_style))
    story.append(Paragraph(
        'This capstone project demonstrates a complete data science workflow from problem definition through data preprocessing, EDA, statistical validation, model development, and actionable business recommendations. '
        'The churn prediction model provides a reliable mechanism for identifying high-risk customers and improving retention strategy effectiveness in the telecom industry.',
        body_style,
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph('Project completed successfully.', styles['Italic']))

    doc.build(story)


def main():
    ensure_directory()
    df = load_and_clean_data()

    summary = calculate_summary_statistics(df)
    churn_plot = save_churn_distribution_plot(df)
    contract_plot = save_contract_analysis_plot(df)
    charges_plot = save_charges_boxplot(df)

    contingency, chi2, p_value = chi_square_analysis(df)
    pipeline, metrics, confusion, top_features, classification = build_model(df)
    save_metrics_csv(metrics, confusion, classification)

    generate_pdf_report(
        df=df,
        summary=summary,
        metrics=metrics,
        confusion=confusion,
        top_features=top_features,
        contingency=contingency,
        chi2=chi2,
        p_value=p_value,
        churn_plot=churn_plot,
        contract_plot=contract_plot,
        charges_plot=charges_plot,
    )

    print('Generated files:')
    print(PDF_PATH)
    print(os.path.join(OUTPUT_DIR, 'model_metrics.csv'))
    print(churn_plot)
    print(contract_plot)
    print(charges_plot)
    print('\nModel metrics:')
    for k, v in metrics.items():
        print(f'{k}: {v}')
    print(f'\nChi-square: {chi2:.4f}, p-value: {p_value:.4e}')


if __name__ == '__main__':
    main()
