#!/usr/bin/env python3
"""Generate the FAQ knowledge base and build a FAISS index.

Run once (or whenever the FAQ corpus changes):
    python build_faq_index.py

Outputs:
    faq/knowledge_base.json   -- human-readable FAQ entries
    faq/faq_metadata.json     -- id/category/question/answer/keywords per entry
    faq/faq_index.faiss       -- FAISS IndexFlatIP over normalized embeddings
"""

from __future__ import annotations

import json
import os
import time

import faiss
import numpy as np

from config import EmbeddingConfig, FAQConfig
from faq.embedder import Embedder

# ──────────────────────────────────────────────────────────────────────
# FAQ Knowledge Base
# ──────────────────────────────────────────────────────────────────────

FAQ_ENTRIES: list[dict] = [
    # ── Credit Cards (8) ─────────────────────────────────────────────
    {
        "id": "CC-001",
        "category": "Credit Cards",
        "question": "What are the eligibility criteria for a credit card?",
        "answer": (
            "To apply for a credit card you must be a Bangladeshi citizen or "
            "resident aged 21-65 with a minimum monthly income of BDT 25,000. "
            "Salaried individuals need at least 6 months of employment with "
            "their current employer, while self-employed applicants must "
            "provide 2 years of business continuity proof. A valid NID and "
            "satisfactory credit history with the Credit Information Bureau "
            "(CIB) are mandatory."
        ),
        "keywords": ["eligibility", "credit card", "apply", "criteria", "income", "NID"],
    },
    {
        "id": "CC-002",
        "category": "Credit Cards",
        "question": "What documents are required to apply for a credit card?",
        "answer": (
            "You will need: (1) completed application form, (2) photocopy of "
            "National ID (NID) or smart card, (3) two recent passport-size "
            "photographs, (4) last 3 months' bank statements, (5) salary "
            "certificate or latest pay slip from employer, (6) TIN certificate, "
            "and (7) utility bill or rental agreement as proof of address. "
            "Self-employed applicants should also provide a trade licence and "
            "latest tax return acknowledgement."
        ),
        "keywords": ["documents", "credit card", "apply", "NID", "TIN", "salary certificate"],
    },
    {
        "id": "CC-003",
        "category": "Credit Cards",
        "question": "I don't have a salary certificate. What alternative proof of income can I provide?",
        "answer": (
            "If a salary certificate is unavailable you may submit any of the "
            "following: bank statements showing regular salary credits for the "
            "last 6 months, an employer-issued income verification letter on "
            "company letterhead, or your latest income tax return with "
            "acknowledgement receipt. Freelancers and gig workers may provide "
            "bKash/Nagad/bank transaction summaries along with contracts or "
            "invoices demonstrating consistent income."
        ),
        "keywords": ["salary proof", "alternative", "income", "freelancer", "bKash"],
    },
    {
        "id": "CC-004",
        "category": "Credit Cards",
        "question": "How long does credit card delivery take after approval?",
        "answer": (
            "Once your credit card application is approved, the card is "
            "typically dispatched within 7-10 working days. Delivery within "
            "Dhaka city usually takes 2-3 working days after dispatch, while "
            "outside Dhaka it may take 5-7 working days via courier. You will "
            "receive an SMS notification with the courier tracking number. "
            "Please ensure someone is available at your registered address to "
            "receive the card, as it requires a signature upon delivery."
        ),
        "keywords": ["delivery", "credit card", "timeline", "dispatch", "courier"],
    },
    {
        "id": "CC-005",
        "category": "Credit Cards",
        "question": "How is my credit card limit determined?",
        "answer": (
            "Your credit limit is assessed based on your gross monthly income, "
            "existing debt obligations, CIB report, and overall banking "
            "relationship. Generally, the limit is set between 2 to 3 times "
            "your net monthly income. You may request a limit enhancement after "
            "6 months of satisfactory card usage by submitting updated income "
            "proof. Secured credit cards backed by FDR can offer limits up to "
            "80%% of the deposit value."
        ),
        "keywords": ["credit limit", "determine", "income", "FDR", "enhancement"],
    },
    {
        "id": "CC-006",
        "category": "Credit Cards",
        "question": "What are the annual fees for credit cards?",
        "answer": (
            "Annual fees vary by card tier: Classic cards BDT 1,500, Gold cards "
            "BDT 3,000, Platinum cards BDT 5,000, and Signature/World cards "
            "BDT 10,000. The first-year annual fee is often waived as a "
            "promotional offer. Subsequent year fees may be waived if you meet "
            "the minimum annual spend threshold, which ranges from BDT 50,000 "
            "for Classic to BDT 3,00,000 for Signature cards. Government VAT "
            "of 15%% applies on all fees."
        ),
        "keywords": ["annual fee", "credit card", "charges", "waiver", "VAT"],
    },
    {
        "id": "CC-007",
        "category": "Credit Cards",
        "question": "How do reward points work on credit cards?",
        "answer": (
            "You earn 1 reward point for every BDT 50 spent on retail "
            "purchases. Dining and online transactions earn double points "
            "(2 per BDT 50). Points are credited to your account within 2 "
            "billing cycles and are valid for 24 months from the date of "
            "earning. You can redeem points through our rewards catalogue for "
            "gift vouchers, air miles, or statement credits. A minimum of "
            "1,000 accumulated points is required for redemption. Cash "
            "advances, balance transfers, and fee payments do not earn points."
        ),
        "keywords": ["reward points", "earn", "redeem", "credit card", "dining"],
    },
    {
        "id": "CC-008",
        "category": "Credit Cards",
        "question": "How do I cancel my credit card?",
        "answer": (
            "To cancel your credit card, first clear all outstanding dues "
            "including any unbilled transactions. Then submit a written "
            "cancellation request at your home branch or call our 24/7 hotline "
            "at 16789. The bank will process the closure within 7 working days "
            "and send a confirmation SMS. Any remaining reward points will be "
            "forfeited upon cancellation. Please destroy the physical card by "
            "cutting through the chip and magnetic strip. A No Liability "
            "Certificate will be issued within 30 days of closure."
        ),
        "keywords": ["cancel", "close", "credit card", "closure", "no liability"],
    },

    # ── Accounts (6) ─────────────────────────────────────────────────
    {
        "id": "AC-001",
        "category": "Accounts",
        "question": "What is the minimum balance requirement for a savings account?",
        "answer": (
            "The minimum average balance for a regular savings account is "
            "BDT 1,000. For premium savings or high-yield accounts the "
            "requirement is BDT 25,000. Students and senior citizens enjoy a "
            "reduced minimum balance of BDT 500. If the account balance falls "
            "below the minimum for two consecutive months, a maintenance "
            "charge of BDT 200 per quarter will be applied. Islamic banking "
            "Mudaraba savings accounts follow the same thresholds."
        ),
        "keywords": ["minimum balance", "savings", "account", "maintenance charge"],
    },
    {
        "id": "AC-002",
        "category": "Accounts",
        "question": "How do I open a new bank account?",
        "answer": (
            "Visit any branch with: (1) original NID or smart card, (2) two "
            "passport-size photographs attested by the introducer, (3) an "
            "introducer who holds an account at the same branch, (4) proof of "
            "address such as a utility bill, (5) TIN certificate if applicable, "
            "and (6) initial deposit of at least BDT 1,000 for savings or "
            "BDT 5,000 for current accounts. The account is usually activated "
            "within the same day. You can also start the process online through "
            "our eKYC portal using NID verification."
        ),
        "keywords": ["open account", "new account", "NID", "introducer", "eKYC"],
    },
    {
        "id": "AC-003",
        "category": "Accounts",
        "question": "What happens if my account becomes dormant?",
        "answer": (
            "An account is classified as dormant if there are no customer-"
            "initiated transactions for 12 consecutive months. Dormant "
            "accounts cannot process debit transactions, standing instructions, "
            "or cheque encashments. To reactivate, visit your home branch with "
            "your original NID and submit a reactivation request form. The "
            "branch will verify your identity, update your KYC records, and "
            "reactivate the account within 24 hours. No penalty is charged for "
            "reactivation, but any accumulated dormancy-period fees may apply."
        ),
        "keywords": ["dormant", "inactive", "reactivate", "account"],
    },
    {
        "id": "AC-004",
        "category": "Accounts",
        "question": "How can I request an account statement?",
        "answer": (
            "You can obtain your account statement through several channels: "
            "(1) Internet banking: download PDF statements for any date range, "
            "(2) Mobile app: view and share up to 6 months of mini-statements, "
            "(3) Branch request: submit a statement request form for official "
            "stamped copies (processing takes 1-2 working days, fee BDT 100 "
            "per page), (4) Email: request monthly e-statements through your "
            "internet banking profile at no charge. Statements for the current "
            "financial year are available online; older statements require a "
            "branch visit."
        ),
        "keywords": ["statement", "account statement", "e-statement", "download"],
    },
    {
        "id": "AC-005",
        "category": "Accounts",
        "question": "What are the rules for opening a joint account?",
        "answer": (
            "A joint account can be opened by two or more individuals who are "
            "at least 18 years old. All applicants must provide original NID, "
            "photographs, and proof of address. The account mandate can be set "
            "as 'Either or Survivor', 'Former or Survivor', or 'Jointly' based "
            "on the signatories' preference. For 'Jointly' operated accounts, "
            "all signatories must authorise every transaction. Each holder "
            "receives a separate debit card. In case of the death of one "
            "holder, the surviving signatory can operate the account based on "
            "the mandate, subject to Bangladesh Bank regulations."
        ),
        "keywords": ["joint account", "mandate", "survivor", "signatory"],
    },
    {
        "id": "AC-006",
        "category": "Accounts",
        "question": "How do I close my bank account?",
        "answer": (
            "To close your account, visit your home branch with your original "
            "NID, cheque book (unused leaves must be returned), debit card, and "
            "a written closure request. Ensure all standing instructions and "
            "auto-debits are cancelled beforehand. Any remaining balance will "
            "be paid via pay-order or transferred to another account. An "
            "account closing fee of BDT 500 applies if the account is closed "
            "within 6 months of opening. The process typically completes "
            "within 3-5 working days."
        ),
        "keywords": ["close account", "closure", "closing fee", "debit card return"],
    },

    # ── Loans (5) ────────────────────────────────────────────────────
    {
        "id": "LN-001",
        "category": "Loans",
        "question": "What are the eligibility criteria for a personal loan?",
        "answer": (
            "To qualify for a personal loan you must be a Bangladeshi national "
            "aged 22-58, with a minimum monthly income of BDT 20,000. Salaried "
            "applicants need at least 1 year of total employment and 6 months "
            "with the current employer. Self-employed applicants must "
            "demonstrate 2 years of business continuity. A clean CIB report is "
            "essential. Loan amounts range from BDT 50,000 to BDT 30,00,000 "
            "with tenures of 12 to 60 months. The debt-to-income ratio must "
            "not exceed 50%% of net monthly income."
        ),
        "keywords": ["personal loan", "eligibility", "income", "CIB", "tenure"],
    },
    {
        "id": "LN-002",
        "category": "Loans",
        "question": "What is the current interest rate for home loans?",
        "answer": (
            "Our home loan interest rate currently starts at 9.00%% per annum "
            "on a reducing balance basis, subject to Bangladesh Bank's lending "
            "rate cap guidelines. The effective rate depends on the loan amount, "
            "tenure, and your risk profile. Loans up to BDT 1 crore typically "
            "attract the base rate, while amounts above BDT 1 crore may carry "
            "a premium of 0.25-0.50%%. The rate is reviewable annually. Fixed-"
            "rate options are available for the first 3 years at a slightly "
            "higher rate of 9.50%%."
        ),
        "keywords": ["home loan", "interest rate", "housing", "mortgage", "reducing balance"],
    },
    {
        "id": "LN-003",
        "category": "Loans",
        "question": "How is my EMI calculated?",
        "answer": (
            "Your Equated Monthly Instalment (EMI) is calculated using the "
            "reducing balance method with the formula: EMI = [P x R x (1+R)^N] "
            "/ [(1+R)^N - 1], where P is the principal, R is the monthly "
            "interest rate, and N is the number of months. For example, a "
            "BDT 10,00,000 loan at 12%% p.a. for 36 months would have an EMI "
            "of approximately BDT 33,214. You can use our EMI calculator on "
            "the mobile app or website to get an exact figure before applying."
        ),
        "keywords": ["EMI", "instalment", "calculation", "reducing balance"],
    },
    {
        "id": "LN-004",
        "category": "Loans",
        "question": "Is there a penalty for prepaying my loan?",
        "answer": (
            "Prepayment is allowed after a minimum lock-in period of 6 months "
            "from the date of first disbursement. For personal loans, a "
            "prepayment penalty of 2%% on the outstanding principal applies. "
            "Home loans carry a 1%% prepayment charge if closed within the "
            "first 3 years; no penalty after that. Auto loans have no "
            "prepayment penalty. Partial prepayments (minimum BDT 50,000) can "
            "be made once per quarter to reduce either the tenure or the EMI "
            "amount, at your discretion."
        ),
        "keywords": ["prepayment", "penalty", "early closure", "partial payment"],
    },
    {
        "id": "LN-005",
        "category": "Loans",
        "question": "What documents are needed to apply for a loan?",
        "answer": (
            "For a loan application, please provide: (1) completed application "
            "form with photograph, (2) NID or smart card (original and copy), "
            "(3) last 6 months' bank statements, (4) salary certificate and "
            "latest 3 pay slips (salaried) or trade licence and 2 years' "
            "audited financials (self-employed), (5) TIN certificate and tax "
            "return acknowledgement, (6) proof of address (utility bill or "
            "rental agreement), and (7) for home loans additionally: property "
            "documents, approved building plan, and valuation report."
        ),
        "keywords": ["loan documents", "application", "NID", "trade licence"],
    },

    # ── Digital Banking (4) ──────────────────────────────────────────
    {
        "id": "DB-001",
        "category": "Digital Banking",
        "question": "How do I set up the mobile banking app?",
        "answer": (
            "Download our official app from Google Play Store or Apple App "
            "Store. Open the app and tap 'Register'. Enter your 10-digit "
            "account number and the mobile number registered with the bank. "
            "You will receive a 6-digit OTP via SMS - enter it within 3 "
            "minutes. Set your login PIN (6 digits) and enable biometric "
            "authentication if your device supports it. For security, the app "
            "can only be active on one device at a time. If you change your "
            "phone, you will need to re-register."
        ),
        "keywords": ["mobile app", "setup", "register", "OTP", "biometric"],
    },
    {
        "id": "DB-002",
        "category": "Digital Banking",
        "question": "How do I activate internet banking?",
        "answer": (
            "Visit your home branch with your NID and submit an internet "
            "banking enrolment form. You will receive a user ID via SMS and "
            "an initial password via registered email within 24 hours. Log in "
            "at our banking portal, change your password on first login, and "
            "set up your security questions. For full transaction capabilities, "
            "collect the hardware token or activate soft-token on the mobile "
            "app. Internet banking gives you access to fund transfers (BEFTN, "
            "NPSB, RTGS), bill payments, and FDR management."
        ),
        "keywords": ["internet banking", "activate", "enrol", "token", "BEFTN"],
    },
    {
        "id": "DB-003",
        "category": "Digital Banking",
        "question": "How do I reset my internet banking or mobile app password?",
        "answer": (
            "For mobile app: tap 'Forgot PIN' on the login screen, verify "
            "your identity with your registered mobile number OTP and date of "
            "birth, then set a new PIN. For internet banking: click 'Forgot "
            "Password', enter your user ID and registered email, answer your "
            "security question, and a temporary password will be sent to your "
            "email. If both methods fail, visit any branch with your NID for "
            "a manual reset. For security, your account will be temporarily "
            "locked after 5 consecutive failed login attempts."
        ),
        "keywords": ["password reset", "forgot PIN", "locked", "OTP"],
    },
    {
        "id": "DB-004",
        "category": "Digital Banking",
        "question": "What are the daily transaction limits for digital banking?",
        "answer": (
            "Default daily limits are: fund transfer via NPSB/BEFTN BDT "
            "5,00,000, RTGS BDT 25,00,000, intra-bank transfer BDT 10,00,000, "
            "bill payments BDT 2,00,000, and mobile top-up BDT 5,000. Limits "
            "can be customised through internet banking up to the maximum "
            "allowed or reduced for additional security. bKash/Nagad wallet "
            "transfers are capped at BDT 25,000 per transaction and BDT 1,00,000 "
            "per day. Changes to transaction limits take effect after OTP "
            "verification and apply from the next business day."
        ),
        "keywords": ["transaction limit", "daily limit", "NPSB", "RTGS", "bKash"],
    },

    # ── Disputes (4) ─────────────────────────────────────────────────
    {
        "id": "DS-001",
        "category": "Disputes",
        "question": "How do I file a transaction dispute?",
        "answer": (
            "You can file a dispute through: (1) Mobile app: go to "
            "Transactions > select the transaction > tap 'Raise Dispute', "
            "(2) Internet banking: navigate to Services > Dispute Centre, "
            "(3) Call centre: dial 16789 and select option 3, or (4) Branch "
            "visit: fill out the dispute form at the service desk. Provide the "
            "transaction date, amount, merchant name, and a description of the "
            "issue. You must file the dispute within 30 days of the transaction "
            "date. A reference number will be provided for tracking."
        ),
        "keywords": ["dispute", "file", "raise", "transaction", "complaint"],
    },
    {
        "id": "DS-002",
        "category": "Disputes",
        "question": "What happens if there is an unauthorized transaction on my account?",
        "answer": (
            "Report unauthorized transactions immediately by calling our 24/7 "
            "fraud hotline at 16789 (option 5) or visiting the nearest branch. "
            "Your card or digital banking access will be blocked instantly to "
            "prevent further misuse. An investigation will be initiated within "
            "24 hours. Under Bangladesh Bank guidelines, if you report within "
            "3 working days and are not at fault, you are eligible for a full "
            "reversal. Investigation typically concludes within 10-45 working "
            "days depending on whether the transaction involved a local or "
            "international merchant."
        ),
        "keywords": ["unauthorized", "fraud", "stolen", "misuse", "reversal"],
    },
    {
        "id": "DS-003",
        "category": "Disputes",
        "question": "How long does a dispute refund take?",
        "answer": (
            "Provisional credit for simple disputes (duplicate charges, failed "
            "ATM withdrawals) is usually applied within 5-7 working days. For "
            "merchant disputes and international chargebacks the timeline is "
            "longer: domestic disputes 15-30 working days, international "
            "disputes up to 45-90 working days depending on the card network "
            "(Visa/Mastercard) arbitration cycle. You will receive SMS updates "
            "at each stage. If the dispute is resolved in your favour, the "
            "provisional credit becomes permanent. If not, the amount will be "
            "re-debited with a written explanation."
        ),
        "keywords": ["refund", "timeline", "dispute", "provisional credit", "chargeback"],
    },
    {
        "id": "DS-004",
        "category": "Disputes",
        "question": "What are the chargeback rules for credit card transactions?",
        "answer": (
            "Chargebacks are governed by Visa/Mastercard network rules. You "
            "may initiate a chargeback for: goods not received, services not "
            "rendered, defective merchandise, duplicate billing, or "
            "unauthorized use. The chargeback request must be filed within 120 "
            "days of the transaction date. The bank submits the claim to the "
            "card network, and the merchant's bank has 30 days to respond. If "
            "the merchant disputes the chargeback, it may enter pre-arbitration "
            "or arbitration, which can take an additional 45-60 days. You are "
            "responsible for providing supporting documents such as receipts, "
            "correspondence with the merchant, and proof of return if applicable."
        ),
        "keywords": ["chargeback", "Visa", "Mastercard", "merchant dispute", "arbitration"],
    },

    # ── General (3) ──────────────────────────────────────────────────
    {
        "id": "GN-001",
        "category": "General",
        "question": "What are the branch operating hours?",
        "answer": (
            "Regular branch hours are Sunday to Thursday, 10:00 AM to 4:00 PM. "
            "Cash counter services are available from 10:00 AM to 3:30 PM. "
            "Branches are closed on Fridays, Saturdays, and government "
            "gazetted holidays. Select flagship branches in Dhaka (Motijheel, "
            "Gulshan, Banani) offer extended hours until 6:00 PM. During "
            "Ramadan, operating hours are typically reduced to 10:00 AM to "
            "3:00 PM. Our 24/7 contact centre and digital banking channels "
            "are always available for urgent matters."
        ),
        "keywords": ["branch hours", "operating hours", "timing", "holiday", "Ramadan"],
    },
    {
        "id": "GN-002",
        "category": "General",
        "question": "What is the daily ATM withdrawal limit?",
        "answer": (
            "The standard daily ATM withdrawal limit is BDT 50,000 per card, "
            "with a maximum of BDT 25,000 per transaction. Premium and "
            "Platinum debit card holders enjoy an enhanced limit of BDT 1,00,000 "
            "per day. For credit card cash advances, the ATM limit is 30%% of "
            "your credit limit or BDT 50,000, whichever is lower. You can "
            "adjust your daily ATM limit downward through the mobile app for "
            "added security. Cash withdrawals from other banks' ATMs incur a "
            "fee of BDT 20 per transaction after the first 5 free monthly "
            "withdrawals."
        ),
        "keywords": ["ATM", "withdrawal limit", "cash", "daily limit"],
    },
    {
        "id": "GN-003",
        "category": "General",
        "question": "How do I contact customer service?",
        "answer": (
            "You can reach us through multiple channels: (1) 24/7 hotline: "
            "16789 from any Bangladeshi mobile or 09612-016789 from abroad, "
            "(2) Email: support@ourbank.com.bd (response within 24 hours), "
            "(3) Live chat on our website and mobile app (available 8 AM - "
            "10 PM), (4) WhatsApp: +880-1XXX-XXXXXX (automated + agent "
            "support), (5) Social media: Facebook page and Twitter handle for "
            "general queries. For account-specific issues, please have your "
            "account number and NID ready for identity verification."
        ),
        "keywords": ["customer service", "contact", "hotline", "helpline", "call centre"],
    },
]


def main() -> None:
    cfg_emb = EmbeddingConfig()
    cfg_faq = FAQConfig()

    # ── ensure output directory exists ────────────────────────────────
    os.makedirs(os.path.dirname(cfg_faq.FAISS_INDEX_PATH), exist_ok=True)

    # ── 1. save knowledge base ────────────────────────────────────────
    print(f"[1/4] Writing {len(FAQ_ENTRIES)} FAQ entries to {cfg_faq.KNOWLEDGE_BASE_PATH}")
    with open(cfg_faq.KNOWLEDGE_BASE_PATH, "w", encoding="utf-8") as f:
        json.dump(FAQ_ENTRIES, f, indent=2, ensure_ascii=False)

    # ── 2. build metadata list (same order as entries) ────────────────
    metadata = [
        {
            "id": e["id"],
            "category": e["category"],
            "question": e["question"],
            "answer": e["answer"],
            "keywords": e["keywords"],
        }
        for e in FAQ_ENTRIES
    ]
    print(f"[2/4] Writing metadata to {cfg_faq.FAQ_METADATA_PATH}")
    with open(cfg_faq.FAQ_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ── 3. embed all questions ────────────────────────────────────────
    print(f"[3/4] Embedding {len(FAQ_ENTRIES)} questions with {cfg_emb.MODEL_NAME}")
    t0 = time.perf_counter()
    embedder = Embedder(model_name=cfg_emb.MODEL_NAME, device=cfg_emb.DEVICE)
    questions = [e["question"] for e in FAQ_ENTRIES]
    vectors = embedder.embed_documents(questions)
    embed_time = time.perf_counter() - t0
    print(f"       Embedding completed in {embed_time:.2f}s "
          f"| vectors shape: {vectors.shape}")

    # ── 4. build FAISS index (inner product on normalised vectors) ────
    print(f"[4/4] Building FAISS IndexFlatIP (dim={cfg_emb.DIMENSION})")
    index = faiss.IndexFlatIP(cfg_emb.DIMENSION)
    index.add(vectors)
    faiss.write_index(index, cfg_faq.FAISS_INDEX_PATH)
    print(f"       Saved index to {cfg_faq.FAISS_INDEX_PATH}")

    # ── stats ─────────────────────────────────────────────────────────
    categories = {}
    for e in FAQ_ENTRIES:
        categories[e["category"]] = categories.get(e["category"], 0) + 1

    print("\n" + "=" * 56)
    print("  FAQ INDEX BUILD COMPLETE")
    print("=" * 56)
    print(f"  Total entries : {len(FAQ_ENTRIES)}")
    print(f"  Dimension     : {cfg_emb.DIMENSION}")
    print(f"  Index type    : IndexFlatIP (cosine via normalised IP)")
    print(f"  Index vectors : {index.ntotal}")
    print(f"  Embed time    : {embed_time:.2f}s")
    print(f"  Category breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat:20s}: {count}")
    print("=" * 56)


if __name__ == "__main__":
    main()
