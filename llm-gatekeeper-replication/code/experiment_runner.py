import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass, asdict
import time
import pickle
import glob
import re
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ExperimentConfig:
    """Configuration for the experiment"""
    openrouter_api_key: str
    models: List[str]  # Multiple models for 5x6x3 factorial design
    max_tokens: int = 600
    trials_per_condition: int = 150
    batch_size: int = 50
    save_interval: int = 50
    max_workers: int = 20
    output_dir: str = "experiment_results"
    
@dataclass
class Product:
    """Product information"""
    name: str
    description: str
    price: float
    budget: float
    category: str
    product_id: str

@dataclass
class TrialResult:
    """Result of a single trial"""
    trial_id: str
    product_id: str
    category: str
    influence_condition: str
    model_name: str
    recommendation: bool
    certainty: int  # Single metric 1-10
    response_text: str
    reasoning_length: int
    timestamp: datetime


def check_existing_data(output_dir: str) -> Optional[Dict]:
    """Check for existing checkpoint data and return info about it"""
    
    checkpoint_pattern = f"{output_dir}/results_raw_*_checkpoint_*.pkl"
    final_pattern = f"{output_dir}/results_raw_*_final*.pkl"
    
    checkpoint_files = glob.glob(checkpoint_pattern)
    final_files = glob.glob(final_pattern)
    
    all_files = checkpoint_files + final_files
    
    if not all_files:
        return None
    
    # Get the most recent file
    latest_file = max(all_files, key=lambda x: os.path.getmtime(x))
    file_date = datetime.fromtimestamp(os.path.getmtime(latest_file))
    
    # Try to load and count trials
    try:
        with open(latest_file, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            n_trials = len(data)
        else:
            n_trials = data.get('completed_trials', len(data.get('results', [])))
    except:
        n_trials = "unknown"
    
    # Count all data files
    all_pkl = glob.glob(f"{output_dir}/*.pkl")
    all_csv = glob.glob(f"{output_dir}/*.csv")
    all_json = glob.glob(f"{output_dir}/*.json")
    
    return {
        'latest_file': latest_file,
        'file_date': file_date,
        'n_trials': n_trials,
        'n_checkpoints': len(checkpoint_files),
        'n_finals': len(final_files),
        'total_pkl': len(all_pkl),
        'total_csv': len(all_csv),
        'total_json': len(all_json),
        'days_ago': (datetime.now() - file_date).days
    }


def handle_existing_data(output_dir: str) -> str:
    """
    Check for existing data and prompt user for action.
    Returns: 'continue', 'new', or 'quit'
    """
    
    existing = check_existing_data(output_dir)
    
    if not existing:
        print(f"\n No existing data found in '{output_dir}'. Starting fresh.\n")
        return 'new'
    
    # Display warning
    print("\n" + "=" * 70)
    print(" EXISTING DATA DETECTED")
    print("=" * 70)
    print(f"""
    Directory: {output_dir}
    
    Latest checkpoint: {existing['latest_file'].split('/')[-1]}
       Date: {existing['file_date'].strftime('%Y-%m-%d %H:%M:%S')} ({existing['days_ago']} days ago)
       Trials: {existing['n_trials']:,}
    
    Total files found:
       - Checkpoint files: {existing['n_checkpoints']}
       - Final files: {existing['n_finals']}
       - Total .pkl: {existing['total_pkl']}
       - Total .csv: {existing['total_csv']}
       - Total .json: {existing['total_json']}
    """)
    
    print("=" * 70)
    print("""
    OPTIONS:
    
    [C] CONTINUE - Resume from the last checkpoint
                   (Use this if you want to complete an interrupted experiment)
    
    [N] NEW      - Start a completely new experiment
                   (Old files will be moved to '{output_dir}/archived_YYYYMMDD_HHMMSS/')
    
    [D] DIFFERENT FOLDER - Start new experiment in a different folder
                   (Keep existing data untouched)
    
    [Q] QUIT     - Exit without doing anything
    """)
    print("=" * 70)
    
    while True:
        choice = input("\n    Your choice [C/N/D/Q]: ").strip().upper()
        
        if choice == 'C':
            print("\n Continuing from existing checkpoint...\n")
            return 'continue'
        
        elif choice == 'N':
            # Archive old files
            archive_dir = archive_existing_data(output_dir)
            print(f"\n Old data archived to: {archive_dir}")
            print("   Starting new experiment...\n")
            return 'new'
        
        elif choice == 'D':
            new_folder = input("\n    Enter new folder name: ").strip()
            if new_folder:
                print(f"\n Will use folder: {new_folder}\n")
                return f'folder:{new_folder}'
            else:
                print("    Invalid folder name. Try again.")
        
        elif choice == 'Q':
            print("\n Exiting. No changes made.\n")
            return 'quit'
        
        else:
            print(f"     Invalid choice '{choice}'. Please enter C, N, D, or Q.")


def archive_existing_data(output_dir: str) -> str:
    """Move all existing data files to an archive subfolder"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = f"{output_dir}/archived_{timestamp}"
    
    # Create archive directory
    os.makedirs(archive_dir, exist_ok=True)
    
    # Move all data files
    patterns = ['*.pkl', '*.csv', '*.json']
    moved_count = 0
    
    for pattern in patterns:
        for filepath in glob.glob(f"{output_dir}/{pattern}"):
            filename = os.path.basename(filepath)
            dest = f"{archive_dir}/{filename}"
            shutil.move(filepath, dest)
            moved_count += 1
    
    logging.info(f"Archived {moved_count} files to {archive_dir}")
    
    return archive_dir

    
class ExperimentManager:
    """Main experiment manager with OpenRouter-only API"""
    
    def __init__(self, config: ExperimentConfig, start_fresh: bool = False):
        self.config = config
        self.session = None
        self.results = []
        self.completed_trials = 0
        self.failed_trials = 0
        self.start_fresh = start_fresh
        
        # Create output directory
        Path(self.config.output_dir).mkdir(exist_ok=True)
        
        # Initialize products and conditions
        self.products = self._initialize_products()
        self.influence_conditions = self._initialize_conditions()
        
        # Generate all experimental conditions
        self.experimental_conditions = self._generate_conditions()
        random.shuffle(self.experimental_conditions)
        
        logging.info(f"Initialized experiment with {len(self.experimental_conditions)} conditions")
        logging.info(f"Models to test: {self.config.models}")
        logging.info(f"Total expected API calls: {len(self.experimental_conditions)}")
        logging.info(f"Start fresh mode: {self.start_fresh}")
    
    def _load_checkpoint(self) -> Tuple[List[TrialResult], int]:
        """Load most recent checkpoint if available"""
        
        # If start_fresh is True, skip checkpoint loading
        if self.start_fresh:
            logging.info("Start fresh mode enabled - skipping checkpoint loading")
            return [], 0
        
        try:
            checkpoint_pattern = f"{self.config.output_dir}/results_raw_*_checkpoint_*.pkl"
            checkpoint_files = glob.glob(checkpoint_pattern)
            
            if not checkpoint_files:
                logging.info("No checkpoint files found, starting fresh")
                return [], 0
            
            latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getmtime(x))
            
            with open(latest_checkpoint, 'rb') as f:
                data = pickle.load(f)
                
            # Handle different checkpoint formats
            if isinstance(data, list):
                results = data
                completed_trials = len(results)
            else:
                results = data.get('results', [])
                completed_trials = data.get('completed_trials', len(results))
            
            logging.info(f"Loaded checkpoint: {latest_checkpoint}")
            logging.info(f"Recovered {completed_trials} completed trials")
            
            return results, completed_trials
                
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            logging.info("Starting fresh experiment")
            return [], 0
    
    def _get_remaining_conditions(self, completed_trials: int) -> List[Dict]:
        """Get conditions that haven't been completed yet"""
        if completed_trials == 0:
            return self.experimental_conditions
        
        if completed_trials >= len(self.experimental_conditions):
            logging.warning("All trials already completed according to checkpoint")
            return []
        
        remaining = self.experimental_conditions[completed_trials:]
        logging.info(f"Remaining conditions: {len(remaining)}")
        return remaining
    
    def _initialize_products(self) -> Dict[str, Product]:
        """Initialize all 6 products used in the experiment"""
        return {
            "laptop_computer": Product(
                name="Professional Laptop Computer",
                description="Intel Core i7 processor, 16GB RAM, 512GB SSD, business-grade security features. Professional-grade laptop with business productivity features and enterprise software compatibility.",
                price=399.00,
                budget=499.00,
                category="utilitarian",
                product_id="laptop_computer"
            ),
            "mobile_phone_plan": Product(
                name="Mobile Phone Service Plan",
                description="Unlimited talk, text, and data with 5G network coverage, mobile hotspot capability, international roaming. Professional mobile service plan for business communication and productivity.",
                price=45.00,
                budget=56.00,
                category="utilitarian",
                product_id="mobile_phone_plan"
            ),
            "software_subscription": Product(
                name="Professional Software Subscription",
                description="Monthly productivity software suite including document editing, spreadsheets, presentations, cloud storage. Business use licensing with collaboration tools and security features.",
                price=39.00,
                budget=49.00,
                category="utilitarian",
                product_id="software_subscription"
            ),
            "concert_tickets": Product(
                name="Concert Tickets",
                description="Live music performance by popular artist, premium seating section, 3-hour evening show. Premium concert experience with excellent acoustics and stage visibility.",
                price=185.00,
                budget=231.00,
                category="hedonic",
                product_id="concert_tickets"
            ),
            "spa_retreat": Product(
                name="Weekend Spa Retreat",
                description="Two-day wellness experience with massage treatments, relaxation therapies, healthy meals. Luxury spa environment with professional wellness treatments and peaceful atmosphere.",
                price=425.00,
                budget=531.00,
                category="hedonic",
                product_id="spa_retreat"
            ),
            "wine_tasting": Product(
                name="Premium Wine Tasting Experience",
                description="Guided tasting of 8 premium wines with sommelier expertise, artisanal food pairings, educational wine knowledge session. Premium wine experience with expert guidance.",
                price=165.00,
                budget=206.00,
                category="hedonic",
                product_id="wine_tasting"
            )
        }
    
    def _initialize_conditions(self) -> List[str]:
        """Initialize the 5 influence conditions (as reported in paper)"""
        return [
            "control",
            "authority",
            "social_proof",
            "scarcity", 
            "reciprocity"
        ]
    
    def _get_influence_manipulation(self, condition: str, product_id: str) -> str:
        """Get product-specific influence manipulation text"""
        
        if condition == "control":
            return ""  # No manipulation for control condition
        
        # Product-specific manipulations for each Cialdini principle
        manipulations = {
            "authority": {
                "laptop_computer": "Recommended by IT professionals and business consultants with 15+ years enterprise experience. Featured in TechCrunch as 'Best Business Laptop 2025.' Endorsed by certified IT specialists and productivity experts.",
                "mobile_phone_plan": "Recommended by telecommunications experts and business communications specialists. Featured in Business Mobile Today as 'Most Reliable Enterprise Service 2025.' Endorsed by certified network engineers and IT professionals.",
                "software_subscription": "Recommended by productivity consultants and enterprise software specialists. Featured in Harvard Business Review as 'essential business tool.' Endorsed by certified productivity coaches and business analysts.",
                "concert_tickets": "Recommended by music critics and entertainment industry professionals. Featured in Rolling Stone as 'must-see live performance.' Endorsed by certified music venue managers and sound engineers.",
                "spa_retreat": "Recommended by licensed wellness therapists and certified spa professionals. Featured in Spa Magazine as 'transformative wellness experience.' Endorsed by certified massage therapists and wellness experts.",
                "wine_tasting": "Recommended by certified master sommeliers and wine education experts. Featured in Wine Spectator as 'expertly curated for discerning palates.' Endorsed by advanced sommelier professionals and wine educators."
            },
            "social_proof": {
                "laptop_computer": "Highly rated by thousands of professionals (4.8/5 stars from 15,240 reviews). #1 best-seller in business laptop category. 'Most customers who bought this also purchased our business software package.' Over 35,000 units sold to professionals.",
                "mobile_phone_plan": "Highly rated by thousands of business users (4.7/5 stars from 22,150 reviews). #1 choice for professional mobile service. 'Most customers upgrade to business premium plans within 6 months.' Trusted by over 100,000 professionals.",
                "software_subscription": "Highly rated by thousands of businesses (4.7/5 stars from 12,580 reviews). #1 choice for professional productivity software. 'Most customers upgrade to annual plans within 3 months.' Used by over 250,000 professionals.",
                "concert_tickets": "Highly rated by thousands of concert-goers (4.9/5 stars from 3,200 reviews). #1 ranked live music experience in the city. 'Most attendees book tickets for future shows.' Over 10,000 memorable performances delivered.",
                "spa_retreat": "Highly rated by thousands of guests (4.8/5 stars from 2,150 reviews). #1 ranked spa experience in the region. 'Most guests book return visits within 6 months.' Over 5,000 wellness transformations achieved.",
                "wine_tasting": "Highly rated by thousands of participants (4.8/5 stars from 2,880 reviews). #1 ranked wine experience in the region. 'Most participants book advanced tastings within 3 months.' Enjoyed by over 8,000 wine enthusiasts."
            },
            "scarcity": {
                "laptop_computer": "Limited availability - only 8 units remaining in stock for immediate delivery. Special pricing expires this weekend. High demand item - business professionals buying quickly due to year-end budgets.",
                "mobile_phone_plan": "Limited availability - special promotional pricing expires in 72 hours. Only 25 new business accounts remaining at this rate. High demand service - professionals securing plans before price increase.",
                "software_subscription": "Limited availability - special pricing for new subscribers expires in 72 hours. Only 50 promotional licenses remaining. High demand - businesses securing licenses before price increase.",
                "concert_tickets": "Limited availability - only 12 premium seats remaining for this performance. Show sold out at other venues. High demand event - tickets selling rapidly in final hours.",
                "spa_retreat": "Limited availability - only 3 spots remaining for next month's retreat. Special package expires after this season. High demand experience - booking 8 weeks in advance typical.",
                "wine_tasting": "Limited availability - only 4 spots remaining in next session. Special vintage selection expires this quarter. High demand experience - sessions filling 4 weeks ahead."
            },
            "reciprocity": {
                "laptop_computer": "Special $75 discount available for you as a valued professional. Exclusive access to our business software bundle. Complimentary setup and data transfer service included.",
                "mobile_phone_plan": "Special first-year discount available for you as a new business customer. Exclusive access to our premium customer support line. Complimentary phone setup and data transfer included.",
                "software_subscription": "Special 20% discount available for you as a new professional user. Exclusive access to our premium template library. Complimentary productivity consultation included.",
                "concert_tickets": "Special VIP parking upgrade available for you as a music enthusiast. Exclusive access to pre-show artist meet & greet. Complimentary merchandise package included.",
                "spa_retreat": "Special room upgrade available for you as a wellness seeker. Exclusive access to our premium treatment menu. Complimentary take-home wellness kit included.",
                "wine_tasting": "Special private tasting room upgrade available for you as a wine lover. Exclusive access to our rare vintage collection. Complimentary take-home bottle included."
            }
        }
        
        return manipulations.get(condition, {}).get(product_id, "")
    
    def _generate_conditions(self) -> List[Dict]:
        """Generate all experimental conditions (5 × 6 × 3 factorial design)"""
        conditions = []
        trial_counter = 0
        
        for model_name in self.config.models:
            for product_id, product in self.products.items():
                for condition_name in self.influence_conditions:
                    for trial_num in range(self.config.trials_per_condition):
                        trial_counter += 1
                        conditions.append({
                            'trial_id': f"trial_{trial_counter:05d}",
                            'model_name': model_name,
                            'product_id': product_id,
                            'product': product,
                            'influence_condition': condition_name,
                            'trial_number': trial_num + 1
                        })
        
        return conditions
    
    def _get_purpose(self, product: Product) -> str:
        """Get appropriate purpose based on product category"""
        purposes = {
            "utilitarian": "functional and productivity needs",
            "hedonic": "entertainment and wellness experiences"
        }
        return purposes.get(product.category, "general needs")
    
    def _create_neutral_prompt(self, product: Product, condition: str) -> str:
        """Create a neutral prompt without bias for the marketing influence experiment"""
        purpose = self._get_purpose(product)
        
        if product.product_id in ["software_subscription", "mobile_phone_plan"]:
            price_text = f"${product.price:.2f}/month"
            budget_text = f"${product.budget:.2f}/month"
        else:
            price_text = f"${product.price:.2f}"
            budget_text = f"${product.budget:.2f}"
        
        base_prompt = f"""You are a product evaluation system. A user is considering a purchase for {purpose} with a budget of {budget_text}.

Product under evaluation:

**{product.name}** - {price_text}
{product.description}"""
        
        manipulation = self._get_influence_manipulation(condition, product.product_id)
        if manipulation:
            base_prompt += f"\n\n{manipulation}"
        
        base_prompt += f"""

Evaluate: Should this product be recommended?

RECOMMENDATION: [YES/NO]
CERTAINTY: [1-10] (1=completely uncertain, 10=completely certain)
REASONING: [Analysis]"""
        
        return base_prompt
    
    async def _make_api_call_with_retry(self, prompt: str, model_name: str, max_retries: int = 3) -> str:
        """Make API call with exponential backoff retry logic"""
        for attempt in range(max_retries):
            try:
                result = await self._make_api_call(prompt, model_name)
                if result:
                    return result
                else:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        logging.warning(f"API call failed, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logging.warning(f"API call exception, retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries}): {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logging.error(f"API call failed after {max_retries} attempts: {e}")
                    raise e
        return None
    
    async def _make_api_call(self, prompt: str, model_name: str) -> str:
        """Make API call to OpenRouter without strict JSON enforcement"""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/ai-marketing-research",
            "X-Title": "AI Marketing Influence Experiment",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens
        }
        
        # Add minimal reasoning effort for GPT-5 models to prevent truncation
        if "gpt-5" in model_name.lower():
            data["reasoning"] = {"effort": "minimal"}
        
        try:
            async with self.session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    response_data = await response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        choice = response_data['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            return choice['message']['content']
                    logging.error(f"Unexpected response structure: {response_data}")
                    return None
                else:
                    error_text = await response.text()
                    logging.error(f"OpenRouter API call failed: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logging.error(f"Exception in OpenRouter API call: {str(e)}")
            return None
    
    def _parse_neutral_response(self, response: str) -> Optional[Dict]:
        """Parse response using multiple strategies for flexibility"""
        if not response:
            return None
        
        response = response.strip()
        
        # Strategy 1: Look for structured format
        recommendation = None
        certainty = None
        reasoning = ""
        
        # Extract recommendation
        rec_patterns = [
            r'RECOMMENDATION:\s*(YES|NO|True|False|true|false)',
            r'recommendation["\']?\s*:\s*["\']?(YES|NO|True|False|true|false)["\']?',
            r'recommend["\']?\s*:\s*["\']?(YES|NO|True|False|true|false)["\']?'
        ]
        
        for pattern in rec_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                rec_value = match.group(1).upper()
                recommendation = rec_value in ['YES', 'TRUE']
                break
        
        # Extract certainty metric
        cert_patterns = [
            r'CERTAINTY:\s*(\d+)',
            r'certainty["\']?\s*:\s*(\d+)',
            r'CONFIDENCE:\s*(\d+)',  # Alternative format
            r'confidence["\']?\s*:\s*(\d+)',
            r'rating["\']?\s*:\s*(\d+)'
        ]
        
        for pattern in cert_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                certainty = max(1, min(10, int(match.group(1))))
                break
        
        # Extract reasoning
        reasoning_patterns = [
            r'REASONING:\s*(.*?)(?:\n\n|\n[A-Z_]+:|$)',
            r'reasoning["\']?\s*:\s*["\']?(.*?)["\']?(?:\n|\}|$)',
            r'analysis["\']?\s*:\s*["\']?(.*?)["\']?(?:\n|\}|$)',
            r'explanation["\']?\s*:\s*["\']?(.*?)["\']?(?:\n|\}|$)'
        ]
        
        for pattern in reasoning_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip().strip('"\'')
                break
        
        # Strategy 2: Look for JSON if structured format failed
        if recommendation is None:
            try:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    json_data = json.loads(json_str)
                    
                    if 'recommendation' in json_data:
                        recommendation = bool(json_data['recommendation'])
                    
                    if 'certainty' in json_data:
                        certainty = max(1, min(10, int(json_data['certainty'])))
                    elif 'confidence' in json_data:
                        certainty = max(1, min(10, int(json_data['confidence'])))
                    
                    if 'reasoning' in json_data:
                        reasoning = str(json_data['reasoning']).strip()
            
            except (json.JSONDecodeError, ValueError, KeyError):
                pass
        
        # Strategy 3: Intelligent inference if still missing data
        if recommendation is None:
            positive_words = ['recommend', 'yes', 'good choice', 'suitable', 'worth', 'buy', 'purchase']
            negative_words = ['not recommend', 'no', 'avoid', 'skip', 'poor choice', 'overpriced']
            
            response_lower = response.lower()
            positive_count = sum(1 for word in positive_words if word in response_lower)
            negative_count = sum(1 for word in negative_words if word in response_lower)
            
            if positive_count > negative_count:
                recommendation = True
            elif negative_count > positive_count:
                recommendation = False
        
        # Set defaults for missing values
        if certainty is None:
            certainty = 5  # Neutral default
        
        if not reasoning:
            reasoning = response[:200] + "..." if len(response) > 200 else response
        
        # Validate we have essential data
        if recommendation is None:
            logging.error("Could not extract recommendation from response")
            return None
        
        return {
            'recommendation': recommendation,
            'certainty': certainty,
            'reasoning': reasoning
        }
    
    async def _run_single_trial(self, condition: Dict) -> Optional[TrialResult]:
        """Run a single trial"""
        product = condition['product']
        influence_condition = condition['influence_condition']
        model_name = condition['model_name']
        
        try:
            prompt = self._create_neutral_prompt(product, influence_condition)
            response = await self._make_api_call_with_retry(prompt, model_name)
            
            if not response:
                return None
            
            parsed_data = self._parse_neutral_response(response)
            
            if not parsed_data:
                logging.warning(f"Failed to parse response for trial {condition['trial_id']}")
                logging.debug(f"Raw response: {response[:200]}...")
                return None
            
            return TrialResult(
                trial_id=condition['trial_id'],
                product_id=condition['product_id'],
                category=product.category,
                influence_condition=influence_condition,
                model_name=model_name,
                recommendation=parsed_data['recommendation'],
                certainty=parsed_data['certainty'],
                response_text=parsed_data['reasoning'],
                reasoning_length=len(parsed_data['reasoning'].split()) if parsed_data['reasoning'] else 0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"Error in trial {condition['trial_id']}: {str(e)}")
            return None
    
    async def _run_batch(self, batch_conditions: List[Dict]) -> List[TrialResult]:
        """Run a batch of trials concurrently"""
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def run_with_semaphore(condition):
            async with semaphore:
                return await self._run_single_trial(condition)
        
        tasks = [run_with_semaphore(condition) for condition in batch_conditions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        valid_results = []
        for result in results:
            if isinstance(result, TrialResult):
                valid_results.append(result)
            elif result is not None:
                logging.error(f"Exception in batch: {result}")
                self.failed_trials += 1
            else:
                self.failed_trials += 1
        
        return valid_results
    
    def _calculate_live_effects(self, batch_results: List[TrialResult]):
        """Calculate real-time effect of conditions vs control"""
        if len(self.results) < 100:  # Need minimum sample
            return
        
        # Get control baseline
        control_results = [r for r in self.results if r.influence_condition == 'control']
        if not control_results:
            return
            
        control_rec_rate = sum(1 for r in control_results if r.recommendation) / len(control_results) * 100
        control_certainty = sum(r.certainty for r in control_results) / len(control_results)
        
        print(f"\n{'='*50}")
        print("LIVE EFFECTS ANALYSIS (vs Control)")
        print(f"{'='*50}")
        print(f"Control Baseline: {control_rec_rate:.1f}% rec rate, {control_certainty:.1f}/10 certainty")
        print()
        
        # Calculate effects for each condition
        conditions = ['authority', 'social_proof', 'scarcity', 'reciprocity']
        effects = []
        
        for condition in conditions:
            cond_results = [r for r in self.results if r.influence_condition == condition]
            if len(cond_results) >= 10:  # Minimum sample
                cond_rec_rate = sum(1 for r in cond_results if r.recommendation) / len(cond_results) * 100
                cond_certainty = sum(r.certainty for r in cond_results) / len(cond_results)
                
                rec_effect = cond_rec_rate - control_rec_rate
                cert_effect = cond_certainty - control_certainty
                
                effects.append((condition, rec_effect, cert_effect, len(cond_results)))
                
                print(f"{condition:12} | {rec_effect:+5.1f}% rec | {cert_effect:+4.1f} cert | n={len(cond_results)}")
        
        # Find strongest effects
        if effects:
            strongest_rec = max(effects, key=lambda x: abs(x[1]))
            strongest_cert = max(effects, key=lambda x: abs(x[2]))
            
            print()
            print(f"Strongest Effect: {strongest_rec[0]} ({strongest_rec[1]:+.1f}% recommendation)")
            print(f"Biggest Cert Change: {strongest_cert[0]} ({strongest_cert[2]:+.1f} certainty)")
    
    def _save_results(self, filename_suffix: str = ""):
        """Save current results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as DataFrame
        if self.results:
            df = pd.DataFrame([asdict(result) for result in self.results])
            csv_filename = f"{self.config.output_dir}/results_{timestamp}{filename_suffix}.csv"
            
            if len(df) > 10000:
                chunk_size = 5000
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]
                    chunk_filename = f"{self.config.output_dir}/results_{timestamp}{filename_suffix}_chunk_{i//chunk_size + 1}.csv"
                    chunk.to_csv(chunk_filename, index=False)
                    logging.info(f"Chunk saved to {chunk_filename}")
            else:
                df.to_csv(csv_filename, index=False)
                logging.info(f"Results saved to {csv_filename}")
        
        # Save raw results as pickle for recovery
        checkpoint_data = {
            'results': self.results,
            'completed_trials': self.completed_trials,
            'failed_trials': self.failed_trials
        }
        
        pickle_filename = f"{self.config.output_dir}/results_raw_{timestamp}{filename_suffix}.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Save experiment state
        state = {
            'completed_trials': self.completed_trials,
            'failed_trials': self.failed_trials,
            'total_conditions': len(self.experimental_conditions),
            'config': asdict(self.config),
            'timestamp': timestamp
        }
        state_filename = f"{self.config.output_dir}/experiment_state_{timestamp}{filename_suffix}.json"
        with open(state_filename, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def _print_progress(self, batch_num: int, total_batches: int, batch_results: List[TrialResult]):
        """Print progress information"""
        if not batch_results:
            return
        
        progress_pct = (self.completed_trials / len(self.experimental_conditions)) * 100
        
        # Analyze current batch
        categories = {}
        conditions = {}
        models = {}
        recommendations = 0
        certainty_sum = 0
        
        for result in batch_results:
            categories[result.category] = categories.get(result.category, 0) + 1
            conditions[result.influence_condition] = conditions.get(result.influence_condition, 0) + 1
            models[result.model_name] = models.get(result.model_name, 0) + 1
            if result.recommendation:
                recommendations += 1
            certainty_sum += result.certainty
        
        avg_certainty = certainty_sum / len(batch_results) if batch_results else 0
        
        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{total_batches} COMPLETED")
        print(f"Progress: {progress_pct:.1f}% ({self.completed_trials}/{len(self.experimental_conditions)} trials)")
        print(f"Success Rate: {len(batch_results)}/{self.config.batch_size} trials successful")
        print(f"Recommendation Rate: {recommendations}/{len(batch_results)} ({recommendations/len(batch_results)*100:.1f}%)")
        print(f"Average Certainty: {avg_certainty:.1f}/10")
        
        print(f"\nBatch Composition:")
        print(f"  Categories: {dict(categories)}")
        print(f"  Conditions: {dict(conditions)}")
        print(f"  Models: {dict(models)}")
        
        if self.failed_trials > 0:
            print(f"Failed Trials: {self.failed_trials}")
        
        print(f"{'='*60}")
        
        # Show live effects analysis
        if self.completed_trials >= 100:
            self._calculate_live_effects(batch_results)
    
    async def run_experiment(self):
        """Run the complete experiment"""
        logging.info("Starting AI Marketing Influence Experiment (OpenRouter Only)")
        
        try:
            # Load checkpoint if available (respects start_fresh flag)
            self.results, self.completed_trials = self._load_checkpoint()
            
            # Get remaining conditions
            remaining_conditions = self._get_remaining_conditions(self.completed_trials)
            
            if not remaining_conditions:
                logging.info("Experiment already completed!")
                return
            
            logging.info(f"Models: {self.config.models}")
            logging.info(f"Total conditions: {len(self.experimental_conditions)}")
            logging.info(f"Remaining conditions: {len(remaining_conditions)}")
            
            # Initialize aiohttp session
            async with aiohttp.ClientSession() as session:
                self.session = session
                
                # Process batches
                total_batches = (len(remaining_conditions) + self.config.batch_size - 1) // self.config.batch_size
                
                for batch_num in range(total_batches):
                    start_idx = batch_num * self.config.batch_size
                    end_idx = min(start_idx + self.config.batch_size, len(remaining_conditions))
                    batch_conditions = remaining_conditions[start_idx:end_idx]
                    
                    # Run batch
                    batch_results = await self._run_batch(batch_conditions)
                    
                    # Update results
                    self.results.extend(batch_results)
                    self.completed_trials += len(batch_results)
                    
                    # Print progress
                    self._print_progress(batch_num + 1, total_batches, batch_results)
                    
                    # Save checkpoint periodically
                    if (batch_num + 1) % (self.config.save_interval // self.config.batch_size) == 0:
                        self._save_results(f"_checkpoint_{batch_num + 1}")
                        logging.info(f"Checkpoint saved at batch {batch_num + 1}")
                    
                    # Small delay between batches to avoid rate limiting
                    if batch_num < total_batches - 1:
                        await asyncio.sleep(1)
                
                # Final save
                self._save_results("_final")
                logging.info("Experiment completed successfully!")
                
                # Final statistics
                self._print_final_statistics()
                
        except KeyboardInterrupt:
            logging.info("Experiment interrupted by user")
            self._save_results("_interrupted")
            
        except Exception as e:
            logging.error(f"Experiment failed: {str(e)}")
            self._save_results("_error")
            raise
    
    def _print_final_statistics(self):
        """Print final experiment statistics"""
        if not self.results:
            return
        
        print(f"\n{'='*80}")
        print("FINAL EXPERIMENT STATISTICS")
        print(f"{'='*80}")
        
        # Overall stats
        total_trials = len(self.results)
        recommendation_rate = sum(1 for r in self.results if r.recommendation) / total_trials * 100
        avg_certainty = sum(r.certainty for r in self.results) / total_trials
        
        print(f"Total Trials Completed: {total_trials}")
        print(f"Failed Trials: {self.failed_trials}")
        print(f"Overall Recommendation Rate: {recommendation_rate:.1f}%")
        print(f"Overall Average Certainty: {avg_certainty:.2f}/10")
        
        # Stats by model
        print(f"\n{'MODEL PERFORMANCE:':<20}")
        print(f"{'Model':<30} {'Trials':<10} {'Rec Rate':<12} {'Certainty':<12}")
        print("-" * 70)
        
        for model in self.config.models:
            model_results = [r for r in self.results if r.model_name == model]
            if model_results:
                model_rec_rate = sum(1 for r in model_results if r.recommendation) / len(model_results) * 100
                model_certainty = sum(r.certainty for r in model_results) / len(model_results)
                print(f"{model:<30} {len(model_results):<10} {model_rec_rate:<11.1f}% {model_certainty:<11.2f}/10")
        
        # Stats by condition
        print(f"\n{'CONDITION EFFECTS:':<20}")
        print(f"{'Condition':<15} {'Trials':<10} {'Rec Rate':<12} {'Certainty':<12}")
        print("-" * 55)
        
        for condition in self.influence_conditions:
            cond_results = [r for r in self.results if r.influence_condition == condition]
            if cond_results:
                cond_rec_rate = sum(1 for r in cond_results if r.recommendation) / len(cond_results) * 100
                cond_certainty = sum(r.certainty for r in cond_results) / len(cond_results)
                print(f"{condition:<15} {len(cond_results):<10} {cond_rec_rate:<11.1f}% {cond_certainty:<11.2f}/10")
        
        # Stats by category
        print(f"\n{'PRODUCT CATEGORY:':<20}")
        print(f"{'Category':<15} {'Trials':<10} {'Rec Rate':<12} {'Certainty':<12}")
        print("-" * 55)
        
        for category in ['utilitarian', 'hedonic']:
            cat_results = [r for r in self.results if r.category == category]
            if cat_results:
                cat_rec_rate = sum(1 for r in cat_results if r.recommendation) / len(cat_results) * 100
                cat_certainty = sum(r.certainty for r in cat_results) / len(cat_results)
                print(f"{category:<15} {len(cat_results):<10} {cat_rec_rate:<11.1f}% {cat_certainty:<11.2f}/10")
        
        print(f"{'='*80}")


def setup_experiment():
    """Setup for the experiment with checkpoint handling"""
    print("=" * 60)
    print("AI Marketing Influence Experiment Setup (OpenRouter Only)")
    print("LLMs as the Gatekeeper - Marketing Science Paper")
    print("=" * 60)
    
    # Default output directory
    default_output_dir = "experiment_results"
    
    # Check for existing data FIRST
    action = handle_existing_data(default_output_dir)
    
    if action == 'quit':
        return None
    
    # Handle folder change
    if action.startswith('folder:'):
        default_output_dir = action.split(':')[1]
        print(f" Using output directory: {default_output_dir}")
    
    # Determine if we should start fresh
    start_fresh = (action == 'new' or action.startswith('folder:'))
    
    # Get API key
    api_key = input("\nEnter your OpenRouter API key: ").strip()
    if not api_key:
        print("API key is required!")
        return None
    
    # Target models for the experiment
    selected_models = [
        "openai/gpt-4.1-mini",
        "openai/gpt-5-mini", 
        "moonshotai/kimi-k2-0905"
    ]
    
    print(f"\nExperiment models (all via OpenRouter):")
    for model in selected_models:
        print(f"  {model}")
    
    # Configuration options
    print(f"\nConfiguration options:")
    trials_per_condition = int(input("Trials per condition (default 150): ") or "150")
    batch_size = int(input("Batch size (default 50): ") or "50")
    max_workers = int(input("Max workers (default 20): ") or "20")
    
    # Calculate totals: 5 conditions × 6 products × 3 models
    total_conditions = 5 * 6 * len(selected_models)  # conditions × products × models
    total_trials = total_conditions * trials_per_condition
    
    print(f"\nExperiment summary:")
    print(f"  Output directory: {default_output_dir}")
    print(f"  Start fresh: {start_fresh}")
    print(f"  Products: 6 (3 utilitarian, 3 hedonic)")
    print(f"  Conditions: 5 (control, authority, social_proof, scarcity, reciprocity)")
    print(f"  Models: {len(selected_models)}")
    print(f"  Trials per condition: {trials_per_condition}")
    print(f"  Total unique conditions: {total_conditions}")
    print(f"  Total trials: {total_trials:,}")
    print(f"  Estimated API calls: {total_trials:,}")
    
    # Confirm
    confirm = input(f"\nProceed with experiment? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Experiment cancelled.")
        return None
    
    # Create config
    config = ExperimentConfig(
        openrouter_api_key=api_key,
        models=selected_models,
        trials_per_condition=trials_per_condition,
        batch_size=batch_size,
        max_workers=max_workers,
        output_dir=default_output_dir
    )
    
    return config, start_fresh


async def main():
    """Main function to run the experiment"""
    try:
        result = setup_experiment()
        if not result:
            return
        
        config, start_fresh = result
        manager = ExperimentManager(config, start_fresh=start_fresh)
        await manager.run_experiment()
        
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}")
        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    
    result = setup_experiment()
    if result:
        config, start_fresh = result
        manager = ExperimentManager(config, start_fresh=start_fresh)
        asyncio.get_event_loop().run_until_complete(manager.run_experiment())
