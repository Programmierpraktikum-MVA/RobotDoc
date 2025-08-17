import re
from typing import List, Tuple, Dict, Any

# Extended emergency symptoms with German translations
EMERGENCY_SYMPTOMS = {
    'high_priority': [
        # English
        'chest pain', 'difficulty breathing', 'severe bleeding',
        'unconsciousness', 'seizure', 'stroke symptoms',
        'severe head injury', 'severe burns', 'anaphylaxis',
        # German
        'brustschmerzen', 'atemnot', 'atemprobleme', 'schwere blutung',
        'bewusstlosigkeit', 'krampfanfall', 'epileptischer anfall',
        'schlaganfall', 'schwere kopfverletzung', 'schwere verbrennungen',
        'anaphylaktischer schock', 'herzinfarkt', 'herzstillstand'
    ],
    'medium_priority': [
        # English
        'high fever', 'severe pain', 'sudden severe headache',
        'sudden vision changes', 'sudden speech problems',
        'sudden weakness', 'severe allergic reaction',
        # German
        'hohes fieber', 'starke schmerzen', 'plötzliche starke kopfschmerzen',
        'plötzliche sehstörungen', 'plötzliche sprachprobleme',
        'plötzliche schwäche', 'schwere allergische reaktion',
        'schwindel', 'übelkeit', 'erbrechen', 'durchfall'
    ]
}

# Extended emergency patterns with German patterns
EMERGENCY_PATTERNS = {
    'high_priority': [
        # English patterns
        r'not breathing',
        r'can\'t breathe',
        r'severe chest pain',
        r'unconscious',
        r'severe bleeding',
        r'stroke',
        r'seizure',
        r'severe head injury',
        r'severe burns',
        r'anaphylaxis',
        # German patterns
        r'kann nicht atmen',
        r'atemnot',
        r'starke brustschmerzen',
        r'bewusstlos',
        r'schwere blutung',
        r'schlaganfall',
        r'krampfanfall',
        r'epileptischer anfall',
        r'schwere kopfverletzung',
        r'schwere verbrennungen',
        r'anaphylaktischer schock',
        r'herzinfarkt',
        r'herzstillstand',
        r'keine luft',
        r'erstickt',
        r'herzschmerzen'
    ],
    'medium_priority': [
        # English patterns
        r'high fever',
        r'severe pain',
        r'sudden severe headache',
        r'sudden vision changes',
        r'sudden speech problems',
        r'sudden weakness',
        r'severe allergic reaction',
        # German patterns
        r'hohes fieber',
        r'starke schmerzen',
        r'plötzliche starke kopfschmerzen',
        r'plötzliche sehstörungen',
        r'plötzliche sprachprobleme',
        r'plötzliche schwäche',
        r'schwere allergische reaktion',
        r'schwindel',
        r'übelkeit',
        r'erbrechen',
        r'durchfall',
        r'fieber über 39',
        r'starke übelkeit'
    ]
}

def detect_emergency_extended(message: str) -> Tuple[bool, str, List[str]]:
    """
    Extended emergency detection with German support
    
    Args:
        message: Input message to analyze
        
    Returns:
        Tuple of (is_emergency, priority, detected_symptoms)
    """
    detected_symptoms = []
    priority = 'none'
    
    # Check high priority symptoms first
    for symptom in EMERGENCY_SYMPTOMS['high_priority']:
        if symptom.lower() in message.lower():
            detected_symptoms.append(symptom)
            priority = 'high_priority'
    
    # Check medium priority symptoms if no high priority found
    if priority == 'none':
        for symptom in EMERGENCY_SYMPTOMS['medium_priority']:
            if symptom.lower() in message.lower():
                detected_symptoms.append(symptom)
                priority = 'medium_priority'
    
    # Check high priority patterns
    if priority == 'none':
        for pattern in EMERGENCY_PATTERNS['high_priority']:
            if re.search(pattern, message.lower()):
                detected_symptoms.append(pattern)
                priority = 'high_priority'
    
    # Check medium priority patterns if no high priority found
    if priority == 'none':
        for pattern in EMERGENCY_PATTERNS['medium_priority']:
            if re.search(pattern, message.lower()):
                detected_symptoms.append(pattern)
                priority = 'medium_priority'
    
    return len(detected_symptoms) > 0, priority, detected_symptoms

def get_emergency_response_extended(priority: str, symptoms: List[str]) -> str:
    """
    Extended emergency response with German support
    
    Args:
        priority: Emergency priority level
        symptoms: List of detected symptoms
        
    Returns:
        Emergency response message
    """
    if priority == 'high_priority':
        return (
            "🚨 NOTFALL ERKANNT! Die beschriebenen Symptome ({}) erfordern möglicherweise "
            "sofortige medizinische Hilfe. Bitte rufen Sie sofort den Notruf (112) an "
            "oder gehen Sie zum nächsten Krankenhaus. Wenn Sie bei jemandem sind, "
            "bitten Sie um sofortige Hilfe."
        ).format(', '.join(symptoms))
    
    elif priority == 'medium_priority':
        return (
            "⚠️ WICHTIG: Die beschriebenen Symptome ({}) sollten bald von einem Arzt "
            "untersucht werden. Bitte suchen Sie so schnell wie möglich einen Arzt auf "
            "oder kontaktieren Sie den ärztlichen Bereitschaftsdienst (116 117)."
        ).format(', '.join(symptoms))
    
    return ""

def analyze_medical_urgency(message: str) -> Dict[str, Any]:
    """
    Comprehensive medical urgency analysis
    
    Args:
        message: Input message to analyze
        
    Returns:
        Dictionary with detailed urgency analysis
    """
    is_emergency, priority, symptoms = detect_emergency_extended(message)
    
    analysis = {
        "is_emergency": is_emergency,
        "priority": priority,
        "symptoms": symptoms,
        "urgency_score": 0,
        "recommendations": [],
        "response": ""
    }
    
    # Calculate urgency score
    if priority == 'high_priority':
        analysis["urgency_score"] = 9
        analysis["recommendations"] = [
            "Sofort Notruf 112 anrufen",
            "Nächste Notaufnahme aufsuchen",
            "Bei Bewusstlosigkeit: Stabile Seitenlage",
            "Bei Herzstillstand: Sofortige Reanimation"
        ]
    elif priority == 'medium_priority':
        analysis["urgency_score"] = 6
        analysis["recommendations"] = [
            "Arzt aufsuchen (innerhalb von 24 Stunden)",
            "Ärztlicher Bereitschaftsdienst: 116 117",
            "Symptome überwachen",
            "Bei Verschlechterung: Notruf 112"
        ]
    else:
        analysis["urgency_score"] = 1
        analysis["recommendations"] = [
            "Symptome beobachten",
            "Bei Verschlechterung: Arzt konsultieren"
        ]
    
    # Generate response
    if is_emergency:
        analysis["response"] = get_emergency_response_extended(priority, symptoms)
    
    return analysis

# Test function
def test_emergency_detection():
    """Test the extended emergency detection system"""
    
    test_cases = [
        "Ich habe starke Brustschmerzen und kann kaum atmen",
        "Was ist Diabetes?",
        "Ich habe einen epileptischen Anfall",
        "Mein Kind hat hohes Fieber über 40 Grad",
        "Ich habe plötzlich starke Kopfschmerzen und Übelkeit",
        "Was ist ein Hämangiom?",
        "Ich kann nicht atmen und habe Herzschmerzen",
        "Ich habe Schwindel und Übelkeit",
        "Was ist Angiotensin?",
        "Ich habe eine schwere allergische Reaktion"
    ]
    
    print("🔍 Testing Extended Emergency Detection:")
    print("=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case}")
        print("-" * 40)
        
        analysis = analyze_medical_urgency(test_case)
        
        print(f"Emergency: {analysis['is_emergency']}")
        print(f"Priority: {analysis['priority']}")
        print(f"Urgency Score: {analysis['urgency_score']}/10")
        print(f"Symptoms: {analysis['symptoms']}")
        
        if analysis['is_emergency']:
            print(f"Response: {analysis['response']}")
            print(f"Recommendations: {analysis['recommendations']}")
        
        print()

if __name__ == "__main__":
    test_emergency_detection() 