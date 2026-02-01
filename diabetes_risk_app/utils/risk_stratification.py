"""
Risk stratification utilities.
Converts probability scores into medical risk levels and recommendations.
"""

def stratify_risk(prob):
    """
    Convert probability score to risk level and recommendation.
    
    Parameters:
    -----------
    prob : float
        Probability of being "At-Risk" (0.0 to 1.0)
    
    Returns:
    --------
    tuple : (risk_level, recommendation)
        risk_level : str
            One of: "Low Risk", "Moderate Risk", "High Risk", "Very High Risk"
        recommendation : str
            Personalized medical recommendation
    """
    if prob < 0.20:
        return (
            "Low Risk",
            "Maintain a healthy lifestyle. No immediate medical intervention required. Continue regular check-ups."
        )
    
    elif prob < 0.40:
        return (
            "Moderate Risk",
            "Consider lifestyle modifications (diet, exercise) and schedule follow-up glucose testing within 6 months. Monitor your health indicators regularly."
        )
    
    elif prob < 0.70:
        return (
            "High Risk",
            "You may be developing insulin resistance. Please take a diabetes screening test (HbA1c or Fasting Plasma Glucose test). Consult with your healthcare provider for personalized advice."
        )
    
    else:
        return (
            "Very High Risk",
            "High likelihood of diabetes. Medical testing is strongly advised immediately. Please schedule an appointment with your healthcare provider for comprehensive diabetes screening and evaluation."
        )

def get_risk_color(risk_level):
    """
    Get color code for risk level (for UI display).
    
    Parameters:
    -----------
    risk_level : str
        Risk level string
    
    Returns:
    --------
    str : Color name or hex code
    """
    color_map = {
        "Low Risk": "green",
        "Moderate Risk": "yellow",
        "High Risk": "orange",
        "Very High Risk": "red"
    }
    return color_map.get(risk_level, "gray")

def get_risk_icon(risk_level):
    """
    Get emoji icon for risk level.
    
    Parameters:
    -----------
    risk_level : str
        Risk level string
    
    Returns:
    --------
    str : Emoji icon
    """
    icon_map = {
        "Low Risk": "✅",
        "Moderate Risk": "⚠️",
        "High Risk": "🔶",
        "Very High Risk": "🔴"
    }
    return icon_map.get(risk_level, "❓")

