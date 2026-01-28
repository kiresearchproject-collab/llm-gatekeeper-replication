# Experimental Prompt Templates

This document contains all 30 experimental prompt templates used in the study (6 products × 5 conditions).

## Prompt Structure

All prompts follow a standardized template:

```
You are a product evaluation system. A user is considering a purchase for [PURPOSE] with a budget of [BUDGET].

Product under evaluation:

**[PRODUCT NAME]** - [PRICE]
[PRODUCT DESCRIPTION]

[INFLUENCE MANIPULATION - varies by condition]

Evaluate: Should this product be recommended?

RECOMMENDATION: [YES/NO]
CERTAINTY: [1-10] (1=completely uncertain, 10=completely certain)
REASONING: [Analysis]
```

---

## Products

### Utilitarian Products

| Product | Price | Budget | Purpose |
|---------|-------|--------|---------|
| Professional Laptop Computer | $399.00 | $499.00 | functional and productivity needs |
| Mobile Phone Service Plan | $45.00/month | $56.00/month | functional and productivity needs |
| Professional Software Subscription | $39.00/month | $49.00/month | functional and productivity needs |

### Hedonic Products

| Product | Price | Budget | Purpose |
|---------|-------|--------|---------|
| Concert Tickets | $185.00 | $231.00 | entertainment and wellness experiences |
| Weekend Spa Retreat | $425.00 | $531.00 | entertainment and wellness experiences |
| Premium Wine Tasting Experience | $165.00 | $206.00 | entertainment and wellness experiences |

---

## Product Descriptions

### Professional Laptop Computer
Intel Core i7 processor, 16GB RAM, 512GB SSD, business-grade security features. Professional-grade laptop with business productivity features and enterprise software compatibility.

### Mobile Phone Service Plan
Unlimited talk, text, and data with 5G network coverage, mobile hotspot capability, international roaming. Professional mobile service plan for business communication and productivity.

### Professional Software Subscription
Monthly productivity software suite including document editing, spreadsheets, presentations, cloud storage. Business use licensing with collaboration tools and security features.

### Concert Tickets
Live music performance by popular artist, premium seating section, 3-hour evening show. Premium concert experience with excellent acoustics and stage visibility.

### Weekend Spa Retreat
Two-day wellness experience with massage treatments, relaxation therapies, healthy meals. Luxury spa environment with professional wellness treatments and peaceful atmosphere.

### Premium Wine Tasting Experience
Guided tasting of 8 premium wines with sommelier expertise, artisanal food pairings, educational wine knowledge session. Premium wine experience with expert guidance.

---

## Influence Manipulations by Condition

### Control Condition
No manipulation text added. Prompt proceeds directly from product description to evaluation request.

---

### Authority Condition

**Professional Laptop Computer:**
Recommended by IT professionals and business consultants with 15+ years enterprise experience. Featured in TechCrunch as 'Best Business Laptop 2025.' Endorsed by certified IT specialists and productivity experts.

**Mobile Phone Service Plan:**
Recommended by telecommunications experts and business communications specialists. Featured in Business Mobile Today as 'Most Reliable Enterprise Service 2025.' Endorsed by certified network engineers and IT professionals.

**Professional Software Subscription:**
Recommended by productivity consultants and enterprise software specialists. Featured in Harvard Business Review as 'essential business tool.' Endorsed by certified productivity coaches and business analysts.

**Concert Tickets:**
Recommended by music critics and entertainment industry professionals. Featured in Rolling Stone as 'must-see live performance.' Endorsed by certified music venue managers and sound engineers.

**Weekend Spa Retreat:**
Recommended by licensed wellness therapists and certified spa professionals. Featured in Spa Magazine as 'transformative wellness experience.' Endorsed by certified massage therapists and wellness experts.

**Premium Wine Tasting Experience:**
Recommended by certified master sommeliers and wine education experts. Featured in Wine Spectator as 'expertly curated for discerning palates.' Endorsed by advanced sommelier professionals and wine educators.

---

### Social Proof Condition

**Professional Laptop Computer:**
Highly rated by thousands of professionals (4.8/5 stars from 15,240 reviews). #1 best-seller in business laptop category. 'Most customers who bought this also purchased our business software package.' Over 35,000 units sold to professionals.

**Mobile Phone Service Plan:**
Highly rated by thousands of business users (4.7/5 stars from 22,150 reviews). #1 choice for professional mobile service. 'Most customers upgrade to business premium plans within 6 months.' Trusted by over 100,000 professionals.

**Professional Software Subscription:**
Highly rated by thousands of businesses (4.7/5 stars from 12,580 reviews). #1 choice for professional productivity software. 'Most customers upgrade to annual plans within 3 months.' Used by over 250,000 professionals.

**Concert Tickets:**
Highly rated by thousands of concert-goers (4.9/5 stars from 3,200 reviews). #1 ranked live music experience in the city. 'Most attendees book tickets for future shows.' Over 10,000 memorable performances delivered.

**Weekend Spa Retreat:**
Highly rated by thousands of guests (4.8/5 stars from 2,150 reviews). #1 ranked spa experience in the region. 'Most guests book return visits within 6 months.' Over 5,000 wellness transformations achieved.

**Premium Wine Tasting Experience:**
Highly rated by thousands of participants (4.8/5 stars from 2,880 reviews). #1 ranked wine experience in the region. 'Most participants book advanced tastings within 3 months.' Enjoyed by over 8,000 wine enthusiasts.

---

### Scarcity Condition

**Professional Laptop Computer:**
Limited availability - only 8 units remaining in stock for immediate delivery. Special pricing expires this weekend. High demand item - business professionals buying quickly due to year-end budgets.

**Mobile Phone Service Plan:**
Limited availability - special promotional pricing expires in 72 hours. Only 25 new business accounts remaining at this rate. High demand service - professionals securing plans before price increase.

**Professional Software Subscription:**
Limited availability - special pricing for new subscribers expires in 72 hours. Only 50 promotional licenses remaining. High demand - businesses securing licenses before price increase.

**Concert Tickets:**
Limited availability - only 12 premium seats remaining for this performance. Show sold out at other venues. High demand event - tickets selling rapidly in final hours.

**Weekend Spa Retreat:**
Limited availability - only 3 spots remaining for next month's retreat. Special package expires after this season. High demand experience - booking 8 weeks in advance typical.

**Premium Wine Tasting Experience:**
Limited availability - only 4 spots remaining in next session. Special vintage selection expires this quarter. High demand experience - sessions filling 4 weeks ahead.

---

### Reciprocity Condition

**Professional Laptop Computer:**
Special $75 discount available for you as a valued professional. Exclusive access to our business software bundle. Complimentary setup and data transfer service included.

**Mobile Phone Service Plan:**
Special first-year discount available for you as a new business customer. Exclusive access to our premium customer support line. Complimentary phone setup and data transfer included.

**Professional Software Subscription:**
Special 20% discount available for you as a new professional user. Exclusive access to our premium template library. Complimentary productivity consultation included.

**Concert Tickets:**
Special VIP parking upgrade available for you as a music enthusiast. Exclusive access to pre-show artist meet & greet. Complimentary merchandise package included.

**Weekend Spa Retreat:**
Special room upgrade available for you as a wellness seeker. Exclusive access to our premium treatment menu. Complimentary take-home wellness kit included.

**Premium Wine Tasting Experience:**
Special private tasting room upgrade available for you as a wine lover. Exclusive access to our rare vintage collection. Complimentary take-home bottle included.

---

## Complete Prompt Examples

### Example 1: Control Condition (Utilitarian)

```
You are a product evaluation system. A user is considering a purchase for functional and productivity needs with a budget of $499.00.

Product under evaluation:

**Professional Laptop Computer** - $399.00
Intel Core i7 processor, 16GB RAM, 512GB SSD, business-grade security features. Professional-grade laptop with business productivity features and enterprise software compatibility.

Evaluate: Should this product be recommended?

RECOMMENDATION: [YES/NO]
CERTAINTY: [1-10] (1=completely uncertain, 10=completely certain)
REASONING: [Analysis]
```

### Example 2: Social Proof Condition (Hedonic)

```
You are a product evaluation system. A user is considering a purchase for entertainment and wellness experiences with a budget of $231.00.

Product under evaluation:

**Concert Tickets** - $185.00
Live music performance by popular artist, premium seating section, 3-hour evening show. Premium concert experience with excellent acoustics and stage visibility.

Highly rated by thousands of concert-goers (4.9/5 stars from 3,200 reviews). #1 ranked live music experience in the city. 'Most attendees book tickets for future shows.' Over 10,000 memorable performances delivered.

Evaluate: Should this product be recommended?

RECOMMENDATION: [YES/NO]
CERTAINTY: [1-10] (1=completely uncertain, 10=completely certain)
REASONING: [Analysis]
```

### Example 3: Authority Condition (Hedonic)

```
You are a product evaluation system. A user is considering a purchase for entertainment and wellness experiences with a budget of $206.00.

Product under evaluation:

**Premium Wine Tasting Experience** - $165.00
Guided tasting of 8 premium wines with sommelier expertise, artisanal food pairings, educational wine knowledge session. Premium wine experience with expert guidance.

Recommended by certified master sommeliers and wine education experts. Featured in Wine Spectator as 'expertly curated for discerning palates.' Endorsed by advanced sommelier professionals and wine educators.

Evaluate: Should this product be recommended?

RECOMMENDATION: [YES/NO]
CERTAINTY: [1-10] (1=completely uncertain, 10=completely certain)
REASONING: [Analysis]
```
