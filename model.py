"""
Phishing Email Detection Model
================================
ML model using Scikit-learn with advanced feature extraction
Classifies emails as Phishing or Safe with high accuracy
"""

import re
import json
import pickle
import numpy as np
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Any

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# COMPREHENSIVE PHISHING DATASET
# ─────────────────────────────────────────────

def get_training_data() -> Tuple[List[str], List[int]]:
    """Generate comprehensive training dataset of phishing and legitimate emails."""

    phishing_emails = [
        # Account alerts / credential harvesting
        "URGENT: Your PayPal account has been limited! Click here immediately to verify your identity or your account will be suspended. http://paypal-secure-login.xyz/verify",
        "Dear Customer, Your bank account has been compromised. Login NOW at http://secure-bankofamerica.fakesite.com/login to prevent unauthorized access.",
        "Your Apple ID was used to sign in on a new device. If this wasn't you, click http://apple-id-verify.tk/confirm to secure your account immediately.",
        "ALERT: Unusual activity detected on your Netflix account. Verify your payment method now: http://netflix-billing-update.xyz/payment",
        "Your Amazon account is on hold! Update your billing information at http://amazon-account-verify.ml/billing or your Prime membership will be cancelled.",
        "Microsoft Security Alert: Your Outlook has been hacked. Reset password immediately: http://microsoft-reset.fakesite.net/password",
        "Congratulations! You've been selected for our exclusive reward. Claim your $1000 Amazon gift card: http://amazon-rewards-claim.xyz/gift",
        "Your DHL package could not be delivered. Pay the $2.99 customs fee here: http://dhl-tracking-fee.ml/payment to receive your package",
        "FINAL NOTICE: Your Social Security Number has been suspended. Call 1-800-555-0123 or click http://ssa-verify.xyz/reinstate immediately",
        "IRS Tax Refund: You are eligible for a $3,200 tax refund. Submit your information at http://irs-refund-claim.tk/apply",
        "Your Dropbox storage is full! Upgrade now to avoid losing your files: http://dropbox-upgrade.fakesite.com/storage",
        "WINNER! Your email was selected in our lottery. Claim your £850,000 prize: http://uk-lottery-claim.xyz/winner",
        "LinkedIn: Someone tried to access your account. Secure it now: http://linkedin-security.fakesite.net/verify",
        "Chase Bank: Your account has been flagged for suspicious activity. Verify at: http://chase-secure.xyz/account-verify",
        "Your WhatsApp account will expire tomorrow. Renew for free: http://whatsapp-renew.ml/activate",
        "URGENT crypto opportunity! Invest $500 get $5000 guaranteed. Limited time: http://crypto-invest-now.xyz/profit",
        "Your Venmo account requires immediate verification. Log in here: http://venmo-verify.tk/secure",
        "Federal Student Aid: Your loan application needs verification. Submit docs: http://studentaid-verify.xyz/apply",
        "Instagram: Your account has been flagged. Verify identity or lose access: http://instagram-secure.fakesite.com/verify",
        "Your computer has a virus! Download our free scanner: http://virus-remove-now.xyz/download",
        "HSBC Security: Unusual login detected. Verify your identity: http://hsbc-secure-login.ml/confirm",
        "Walmart Gift Card Winner! You have won $500. Claim now: http://walmart-winner.fakesite.net/claim",
        "Your Google account is at risk! Update security now: http://google-security-update.xyz/verify",
        "FedEx: Your package is held. Pay clearance fee: http://fedex-clearance.tk/payment",
        "Wells Fargo Alert: Card temporarily suspended. Verify: http://wellsfargo-verify.xyz/card",
        "Your Spotify Premium has expired. Renew to keep listening: http://spotify-renew.fakesite.com/billing",
        "COVID-19 Relief Fund: You qualify for $1,400. Apply: http://covid-relief-fund.xyz/apply",
        "Your Twitter/X account will be suspended unless you verify: http://twitter-verify.ml/secure",
        "Investment opportunity: 300% returns guaranteed! Click: http://guaranteed-investment.xyz/profit",
        "Zoom: Your account has suspicious activity. Secure it: http://zoom-secure.fakesite.net/verify",

        # More sophisticated phishing
        "Dear valued customer, we noticed you haven't updated your security information. As part of our enhanced security measures, please verify your account at http://secure-update.xyz/login?token=abc123 within 24 hours.",
        "Your recent order #38291-A has shipped but we need address confirmation. Update here: http://shipping-confirm.fakesite.com/order?id=38291",
        "This is an automated message from IT Support. Your password expires in 1 hour. Reset immediately: http://company-portal.xyz/reset-password",
        "You have a pending wire transfer of $8,500. Confirm or cancel here: http://wire-transfer-bank.ml/confirm?ref=WT8500",
        "Your health insurance benefits are expiring. Re-enroll before deadline: http://insurance-enrollment.fakesite.net/renew",
    ]

    legitimate_emails = [
        # Professional business emails
        "Hi team, please find the meeting agenda for tomorrow's quarterly review attached. We'll be discussing Q3 results and planning for Q4. Please review the documents beforehand.",
        "Dear John, Thank you for your application to our software engineering position. We'd like to schedule an interview for next Tuesday at 2 PM. Please confirm your availability.",
        "Hello, I wanted to follow up on our discussion from last week's conference. I've attached the whitepaper we mentioned. Looking forward to our collaboration.",
        "Team, the deployment to production is scheduled for this Saturday at 2 AM. Please ensure all your code reviews are completed by Friday 5 PM. Thank you.",
        "Hi Sarah, Could you please send me the Q3 financial report when you get a chance? I need it for the board presentation on Thursday. Thanks!",
        "Good morning, This is a reminder that your subscription to Adobe Creative Suite will renew on the 15th. No action needed unless you wish to cancel.",
        "Dear valued customer, Your monthly statement for October is now available in your online banking portal. Log in at www.bankofamerica.com to view your statement.",
        "Hi there, Your recent purchase order #12345 has been confirmed. Expected delivery: 3-5 business days. Track your order at ups.com with tracking number 1Z999AA10123456784.",
        "Hello Dr. Smith, Your appointment is scheduled for November 15th at 3:00 PM. Please arrive 15 minutes early. Call us at (555) 123-4567 if you need to reschedule.",
        "Team, I'm pleased to announce that we've successfully closed the Series B funding round. Details will be shared in tomorrow's all-hands meeting. Great work everyone!",
        "Hi, Thanks for reaching out about our enterprise plan. I've attached our pricing guide and case studies. Happy to schedule a demo at your convenience.",
        "Dear Student, Your assignment submission has been received. Grades will be posted by end of next week. Please check the course portal for feedback.",
        "Hi Mike, Here's the code review feedback for your pull request. Overall great work! A few minor suggestions in the comments. Merge looks good to go.",
        "Good afternoon, The company picnic is scheduled for July 4th weekend. Please RSVP through the HR portal by June 15th. Family members are welcome!",
        "Dear Subscriber, Your New York Times digital subscription is active. Enjoy unlimited access to all articles at nytimes.com. Manage your subscription in account settings.",
        "Hi Jennifer, Attached is the contract for the new office lease. Please review with your legal team and let us know if you have any questions.",
        "Hello, This is a reminder that your library books are due on November 20th. Return or renew online at library.org or call (555) 789-0123.",
        "Team Update: We're migrating our servers this weekend. Please save all work before 6 PM Friday. Systems will be back online by Monday morning.",
        "Dear Parent, This is a reminder about the parent-teacher conference on November 18th from 4-7 PM. Sign up for your time slot at school.edu/conference",
        "Hi, Your GitHub Actions workflow completed successfully. All 47 tests passed. Deployment to staging environment is ready for review.",
        "Good morning, As a valued member, you're eligible for our annual wellness benefit of $500. Submit receipts through the benefits portal at company.com/benefits",
        "Dear Mr. Johnson, Please find enclosed your tax documents for the fiscal year 2024. These have been prepared by our accounting team for your records.",
        "Hi team, Sprint review is tomorrow at 10 AM. Please have your demo environments ready. Stakeholders will be joining remotely via the meeting link sent earlier.",
        "Hello, Your electricity bill for this month is $127.45. Due date: November 30th. Pay online at utility.com, by phone, or at any authorized payment location.",
        "Dear Alumni, You're invited to join our annual fundraising campaign. Your contribution helps support student scholarships. Donate at university.edu/donate",
        "Hi, I'm writing to confirm our consulting engagement starting December 1st. I've attached the SOW and project timeline for your review and signature.",
        "Team, Please complete the mandatory cybersecurity training by end of month. Access the course at training.company.com using your employee credentials.",
        "Good news! Your loan application has been approved. A loan officer will contact you within 2 business days to discuss the terms and next steps.",
        "Hi, This is your weekly digest from LinkedIn. You have 3 connection requests, 12 profile views this week, and 5 new messages waiting for you.",
        "Dear Customer, Your Amazon order has been delivered to your front door. Leave a review to help other customers. Return within 30 days if needed.",
        "Hello, The minutes from last week's board meeting are now available in the shared drive. Please review before our next meeting on the 25th.",
        "Hi, Your flight booking for December 15th is confirmed. Check in online 24 hours before departure at delta.com. Have a great trip!",
        "Team, Year-end performance reviews are due by December 15th. Complete your self-assessment in Workday and schedule your review meeting with your manager.",
        "Dear Professor, I wanted to share my research progress for the semester. I've been working on the topics we discussed and have some interesting findings to present.",
        "Hi, Following up on my job application submitted last week for the Data Scientist role. I'm very interested in the position and would love to discuss further.",
    ]

    emails = phishing_emails + legitimate_emails
    labels = [1] * len(phishing_emails) + [0] * len(legitimate_emails)

    return emails, labels


# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

class EmailFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts numerical features from email text."""

    # Phishing keywords with weights
    PHISHING_KEYWORDS = [
        'urgent', 'immediately', 'suspended', 'verify', 'confirm', 'limited',
        'expire', 'winner', 'congratulations', 'free', 'claim', 'reward',
        'prize', 'selected', 'alert', 'warning', 'unusual', 'suspicious',
        'unauthorized', 'compromised', 'hacked', 'locked', 'blocked',
        'update', 'required', 'action', 'click here', 'login now',
        'final notice', 'last chance', 'act now', 'risk', 'threat',
        'bitcoin', 'crypto', 'investment', 'guaranteed', 'profit',
        'refund', 'tax', 'irs', 'social security', 'bank account',
        'credit card', 'billing', 'payment', 'invoice', 'overdue',
        'password', 'reset', 'secure', 'validate', 'authenticate',
    ]

    SAFE_KEYWORDS = [
        'meeting', 'schedule', 'agenda', 'attached', 'please find',
        'follow up', 'discussion', 'conference', 'team', 'project',
        'review', 'feedback', 'report', 'update', 'reminder',
        'appointment', 'invoice attached', 'regards', 'sincerely',
        'best regards', 'thank you', 'thanks', 'hi team', 'good morning',
        'newsletter', 'unsubscribe', 'manage preferences',
    ]

    SUSPICIOUS_TLDS = [
        '.xyz', '.tk', '.ml', '.ga', '.cf', '.gq', '.top',
        '.click', '.link', '.download', '.work', '.loan',
        '.win', '.review', '.racing', '.stream', '.webcam',
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([self._extract(email) for email in X])

    def _extract(self, text: str) -> List[float]:
        text_lower = text.lower()
        features = []

        # ── URL features ──
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        features.append(len(urls))                                    # url count
        features.append(1 if urls else 0)                             # has url
        features.append(sum(len(u) for u in urls) / max(len(urls), 1))  # avg url length

        # Suspicious TLDs
        suspicious_tld_count = sum(
            1 for url in urls
            for tld in self.SUSPICIOUS_TLDS
            if tld in url.lower()
        )
        features.append(suspicious_tld_count)

        # IP address in URL
        ip_in_url = sum(1 for url in urls if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url))
        features.append(ip_in_url)

        # Brand impersonation
        brands = ['paypal', 'amazon', 'apple', 'microsoft', 'google', 'netflix',
                  'facebook', 'instagram', 'bank', 'chase', 'wells', 'irs',
                  'fedex', 'dhl', 'ups', 'ebay', 'twitter', 'linkedin']
        brand_in_url = sum(
            1 for url in urls
            for brand in brands
            if brand in url.lower() and brand not in urlparse(url).netloc.lower()
        )
        features.append(brand_in_url)

        # Hyphens in domain (suspicious)
        hyphen_domains = sum(
            1 for url in urls
            if urlparse(url).netloc.count('-') > 1
        )
        features.append(hyphen_domains)

        # ── Text / content features ──
        features.append(len(text))                                    # email length
        features.append(len(text.split()))                            # word count
        features.append(len(re.findall(r'[A-Z]', text)))             # uppercase letters
        features.append(text.count('!'))                              # exclamation marks
        features.append(text.count('?'))                              # question marks
        features.append(text.count('$'))                              # dollar signs
        features.append(len(re.findall(r'\d+', text)))                # number sequences

        # Phishing keyword count
        phishing_count = sum(1 for kw in self.PHISHING_KEYWORDS if kw in text_lower)
        features.append(phishing_count)

        # Safe keyword count
        safe_count = sum(1 for kw in self.SAFE_KEYWORDS if kw in text_lower)
        features.append(safe_count)

        # Keyword ratio
        total_kw = phishing_count + safe_count
        features.append(phishing_count / max(total_kw, 1))

        # ── Urgency indicators ──
        urgency_words = ['urgent', 'immediately', 'now', 'asap', 'today',
                         'hours', 'minutes', 'expire', 'deadline', 'final']
        features.append(sum(1 for w in urgency_words if w in text_lower))

        # ── Sender/subject patterns ──
        features.append(1 if re.search(r'dear (customer|user|member|valued)', text_lower) else 0)
        features.append(1 if re.search(r'(won|winner|selected|chosen|lucky)', text_lower) else 0)
        features.append(1 if re.search(r'(account|password|security|verify)', text_lower) else 0)
        features.append(1 if re.search(r'\$[\d,]+', text) else 0)    # money amounts
        features.append(1 if re.search(r'£[\d,]+', text) else 0)     # pound amounts
        features.append(1 if re.search(r'(free|gift|prize|reward)', text_lower) else 0)

        # ── HTML/structural ──
        features.append(text.count('<a '))                            # hyperlink tags
        features.append(text.count('<img'))                           # image tags
        features.append(1 if 'unsubscribe' in text_lower else 0)     # unsubscribe link
        features.append(1 if 'privacy policy' in text_lower else 0)  # privacy policy

        # ── Ratio features ──
        words = text.split()
        cap_ratio = sum(1 for w in words if w.isupper() and len(w) > 2) / max(len(words), 1)
        features.append(cap_ratio)

        digit_ratio = sum(1 for c in text if c.isdigit()) / max(len(text), 1)
        features.append(digit_ratio)

        special_ratio = sum(1 for c in text if not c.isalnum() and c != ' ') / max(len(text), 1)
        features.append(special_ratio)

        return features

    def get_feature_names(self) -> List[str]:
        return [
            'url_count', 'has_url', 'avg_url_length', 'suspicious_tld_count',
            'ip_in_url', 'brand_impersonation', 'hyphen_domains',
            'email_length', 'word_count', 'uppercase_letters',
            'exclamation_marks', 'question_marks', 'dollar_signs', 'number_sequences',
            'phishing_keyword_count', 'safe_keyword_count', 'phishing_keyword_ratio',
            'urgency_indicators', 'generic_greeting', 'winner_language',
            'security_language', 'money_amount', 'pound_amount', 'free_prize_language',
            'hyperlink_tags', 'image_tags', 'has_unsubscribe', 'has_privacy_policy',
            'capital_word_ratio', 'digit_ratio', 'special_char_ratio',
        ]


# ─────────────────────────────────────────────
# PHISHING DETECTOR MODEL
# ─────────────────────────────────────────────

class PhishingDetector:
    def __init__(self):
        self.feature_extractor = EmailFeatureExtractor()
        self.tfidf = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            stop_words='english',
            sublinear_tf=True,
            min_df=1,
        )
        self.scaler = StandardScaler()

        # Ensemble of classifiers
        self.rf  = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
        self.gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        self.lr  = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        self.svm = SVC(kernel='rbf', probability=True, random_state=42)

        self.ensemble = VotingClassifier(
            estimators=[('rf', self.rf), ('gb', self.gb), ('lr', self.lr)],
            voting='soft',
            weights=[3, 2, 1],
        )

        self.is_trained = False
        self.metrics = {}
        self.feature_importances = {}

    def _combine_features(self, emails: List[str], fit: bool = False):
        """Combine TF-IDF and numerical features."""
        if fit:
            tfidf_features = self.tfidf.fit_transform(emails).toarray()
            num_features = self.feature_extractor.fit_transform(emails)
            num_features_scaled = self.scaler.fit_transform(num_features)
        else:
            tfidf_features = self.tfidf.transform(emails).toarray()
            num_features = self.feature_extractor.transform(emails)
            num_features_scaled = self.scaler.transform(num_features)

        return np.hstack([tfidf_features, num_features_scaled])

    def train(self, emails: List[str], labels: List[int]) -> Dict:
        """Train the model and return comprehensive metrics."""
        print("[*] Preparing features...")
        X = self._combine_features(emails, fit=True)
        y = np.array(labels)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("[*] Training ensemble model...")
        self.ensemble.fit(X_train, y_train)
        self.is_trained = True

        # Predictions
        y_pred      = self.ensemble.predict(X_test)
        y_pred_prob = self.ensemble.predict_proba(X_test)[:, 1]

        # Metrics
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.ensemble, X_train, y_train, cv=cv, scoring='accuracy')

        self.metrics = {
            "accuracy":         round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision":        round(precision_score(y_test, y_pred) * 100, 2),
            "recall":           round(recall_score(y_test, y_pred) * 100, 2),
            "f1_score":         round(f1_score(y_test, y_pred) * 100, 2),
            "roc_auc":          round(roc_auc_score(y_test, y_pred_prob) * 100, 2),
            "cv_mean":          round(cv_scores.mean() * 100, 2),
            "cv_std":           round(cv_scores.std() * 100, 2),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, target_names=['Safe', 'Phishing'], output_dict=True
            ),
            "total_samples":    len(emails),
            "phishing_samples": int(sum(labels)),
            "safe_samples":     int(len(labels) - sum(labels)),
            "trained_at":       datetime.now().isoformat(),
        }

        # Feature importances from Random Forest
        rf_fitted = self.ensemble.estimators_[0]
        n_tfidf = len(self.tfidf.get_feature_names_out())
        num_feat_names = self.feature_extractor.get_feature_names()
        importances = rf_fitted.feature_importances_
        num_importances = importances[n_tfidf:]

        self.feature_importances = {
            name: round(float(imp) * 100, 4)
            for name, imp in zip(num_feat_names, num_importances)
        }

        print(f"[+] Training complete! Accuracy: {self.metrics['accuracy']}%")
        return self.metrics

    def predict(self, email: str) -> Dict:
        """Predict whether an email is phishing or safe."""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")

        X = self._combine_features([email], fit=False)
        pred       = self.ensemble.predict(X)[0]
        proba      = self.ensemble.predict_proba(X)[0]

        # Extract features for explanation
        features   = self.feature_extractor._extract(email)
        feat_names = self.feature_extractor.get_feature_names()
        feat_dict  = dict(zip(feat_names, features))

        # Find triggered indicators
        indicators = self._get_indicators(email, feat_dict)

        # Confidence level
        confidence = float(max(proba) * 100)
        conf_level = "Very High" if confidence > 95 else \
                     "High"      if confidence > 85 else \
                     "Medium"    if confidence > 70 else "Low"

        return {
            "prediction":        "Phishing" if pred == 1 else "Safe",
            "is_phishing":       bool(pred == 1),
            "confidence":        round(confidence, 2),
            "confidence_level":  conf_level,
            "phishing_prob":     round(float(proba[1]) * 100, 2),
            "safe_prob":         round(float(proba[0]) * 100, 2),
            "risk_score":        round(float(proba[1]) * 100, 1),
            "indicators":        indicators,
            "feature_values":    feat_dict,
            "analyzed_at":       datetime.now().isoformat(),
        }

    def _get_indicators(self, text: str, features: Dict) -> List[Dict]:
        """Generate human-readable threat indicators."""
        indicators = []
        text_lower = text.lower()

        urls = re.findall(r'http[s]?://\S+', text)

        if features.get('url_count', 0) > 0:
            indicators.append({
                "type": "URL Detected",
                "severity": "medium",
                "detail": f"{int(features['url_count'])} URL(s) found in email",
                "icon": "🔗"
            })

        if features.get('suspicious_tld_count', 0) > 0:
            indicators.append({
                "type": "Suspicious Domain",
                "severity": "critical",
                "detail": "URL uses high-risk TLD (.xyz, .tk, .ml, etc.)",
                "icon": "🚨"
            })

        if features.get('brand_impersonation', 0) > 0:
            indicators.append({
                "type": "Brand Impersonation",
                "severity": "critical",
                "detail": "Known brand name found in suspicious URL",
                "icon": "🎭"
            })

        if features.get('ip_in_url', 0) > 0:
            indicators.append({
                "type": "IP Address in URL",
                "severity": "high",
                "detail": "URL contains raw IP address instead of domain",
                "icon": "⚠️"
            })

        if features.get('urgency_indicators', 0) >= 3:
            indicators.append({
                "type": "Urgency Tactics",
                "severity": "high",
                "detail": f"{int(features['urgency_indicators'])} urgency words detected",
                "icon": "⏰"
            })

        if features.get('phishing_keyword_count', 0) >= 3:
            indicators.append({
                "type": "Phishing Keywords",
                "severity": "high",
                "detail": f"{int(features['phishing_keyword_count'])} phishing keywords found",
                "icon": "🎣"
            })

        if features.get('winner_language', 0):
            indicators.append({
                "type": "Prize/Winner Language",
                "severity": "high",
                "detail": "Email claims you've won something",
                "icon": "🎰"
            })

        if features.get('generic_greeting', 0):
            indicators.append({
                "type": "Generic Greeting",
                "severity": "medium",
                "detail": "Uses impersonal greeting (Dear Customer/User)",
                "icon": "👤"
            })

        if features.get('money_amount', 0) or features.get('pound_amount', 0):
            indicators.append({
                "type": "Financial Lure",
                "severity": "medium",
                "detail": "Email mentions specific monetary amounts",
                "icon": "💰"
            })

        if features.get('capital_word_ratio', 0) > 0.1:
            indicators.append({
                "type": "Excessive Capitals",
                "severity": "low",
                "detail": f"High ratio of ALL-CAPS words detected",
                "icon": "📢"
            })

        if features.get('exclamation_marks', 0) > 2:
            indicators.append({
                "type": "Multiple Exclamations",
                "severity": "low",
                "detail": f"{int(features['exclamation_marks'])} exclamation marks found",
                "icon": "❗"
            })

        if not indicators:
            indicators.append({
                "type": "No Threats Detected",
                "severity": "safe",
                "detail": "Email appears legitimate based on all checks",
                "icon": "✅"
            })

        return indicators

    def save(self, path: str = "phishing_model.pkl"):
        """Save trained model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"[+] Model saved to {path}")

    @staticmethod
    def load(path: str = "phishing_model.pkl") -> 'PhishingDetector':
        """Load model from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ─────────────────────────────────────────────
# BATCH ANALYSIS
# ─────────────────────────────────────────────

def analyze_batch(detector: PhishingDetector, emails: List[str]) -> Dict:
    """Analyze multiple emails at once."""
    results = []
    phishing_count = 0

    for i, email in enumerate(emails):
        result = detector.predict(email)
        result['email_preview'] = email[:100] + '...' if len(email) > 100 else email
        result['email_index'] = i
        results.append(result)
        if result['is_phishing']:
            phishing_count += 1

    return {
        "total": len(emails),
        "phishing_count": phishing_count,
        "safe_count": len(emails) - phishing_count,
        "phishing_rate": round(phishing_count / max(len(emails), 1) * 100, 1),
        "results": results,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def train_and_save() -> PhishingDetector:
    print("=" * 55)
    print("  PHISHING EMAIL DETECTION MODEL — Training")
    print("=" * 55)

    detector = PhishingDetector()
    emails, labels = get_training_data()

    print(f"[*] Dataset: {len(emails)} emails ({sum(labels)} phishing, {len(labels)-sum(labels)} safe)")
    metrics = detector.train(emails, labels)

    print(f"\n{'─'*40}")
    print(f"  Accuracy  : {metrics['accuracy']}%")
    print(f"  Precision : {metrics['precision']}%")
    print(f"  Recall    : {metrics['recall']}%")
    print(f"  F1 Score  : {metrics['f1_score']}%")
    print(f"  ROC AUC   : {metrics['roc_auc']}%")
    print(f"  CV Score  : {metrics['cv_mean']}% ± {metrics['cv_std']}%")
    print(f"{'─'*40}\n")

    detector.save("phishing_model.pkl")
    return detector


if __name__ == "__main__":
    detector = train_and_save()

    test_emails = [
        "URGENT: Your PayPal account has been limited! Verify NOW: http://paypal-secure.xyz/verify",
        "Hi team, please review the attached agenda for tomorrow's meeting. See you at 10 AM!",
        "Congratulations! You've won $1,000,000 lottery. Claim at http://winner-claim.tk/prize",
        "Your Amazon order has been shipped. Track at amazon.com/orders. Expected: 3-5 days.",
    ]

    print("─" * 55)
    print("  SAMPLE PREDICTIONS")
    print("─" * 55)
    for email in test_emails:
        result = detector.predict(email)
        icon = "🚨" if result['is_phishing'] else "✅"
        print(f"{icon} [{result['prediction']:8}] {result['confidence']:.1f}% | {email[:60]}...")
