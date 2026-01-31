"""
NOC (National Occupational Classification) Web Scraper - Enhanced Version
Scrapes all NOC codes, titles, descriptions, hierarchy, and detailed profiles from the Government of Canada website.
"""

import json
import csv
import time
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import pandas as pd


class NOCScraper:
    def __init__(self, scrape_profiles=True):
        self.base_url = "https://noc.esdc.gc.ca/Structure/Hierarchy"
        self.noc_data = []
        self.errors = []
        self.scrape_profiles = scrape_profiles  # Whether to scrape detailed profiles
        
    def scrape(self, headless=True):
        """Main scraping method"""
        print("🚀 Starting NOC Scraper (Enhanced)...")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            page = context.new_page()
            
            try:
                # Navigate to the page
                print(f"\n📡 Loading {self.base_url}...")
                page.goto(self.base_url, wait_until='domcontentloaded', timeout=90000)
                
                # Wait for the structure to load
                print("⏳ Waiting for content to load...")
                time.sleep(5)  # Wait for dynamic content to load
                
                # Try multiple selectors
                try:
                    page.wait_for_selector('.panel-group, .panel, .collapse', timeout=30000)
                except:
                    print("   ⚠️  Panels not found, continuing anyway...")
                
                time.sleep(3)  # Additional wait for dynamic content
                
                # Expand all sections to load the hierarchy
                print("📂 Expanding all sections...")
                self._expand_all_sections(page)
                
                # Extract the hierarchy
                print("🔍 Extracting NOC data...")
                self._extract_hierarchy(page)
                
                # Extract detailed profiles if enabled
                if self.scrape_profiles and len(self.noc_data) > 0:
                    print(f"\n📋 Extracting detailed profiles for {len(self.noc_data)} NOC entries...")
                    self._extract_all_profiles(page)
                
                print(f"\n✅ Successfully scraped {len(self.noc_data)} NOC entries")
                
            except Exception as e:
                print(f"\n❌ Error during scraping: {str(e)}")
                self.errors.append(f"Main scraping error: {str(e)}")
            
            finally:
                browser.close()
        
        return self.noc_data
    
    def _expand_all_sections(self, page):
        """Expand all collapsible sections to reveal the full hierarchy"""
        try:
            # Click "Expand all" button if available
            expand_button = page.locator('text=Expand all').first
            if expand_button.is_visible(timeout=5000):
                expand_button.click()
                time.sleep(3)  # Wait for expansion
                print("   ✓ Clicked 'Expand all' button")
            else:
                # Manually expand each broad category
                print("   ⚙️  Manually expanding categories...")
                broad_categories = page.locator('.panel-heading a[data-toggle="collapse"]').all()
                print(f"   Found {len(broad_categories)} broad categories")
                
                for i, category in enumerate(broad_categories):
                    try:
                        if category.is_visible():
                            category.click()
                            time.sleep(0.5)
                    except:
                        pass
                
                # Wait for all content to load
                time.sleep(2)
                
        except Exception as e:
            print(f"   ⚠️  Could not expand all sections: {str(e)}")
    
    def _extract_hierarchy(self, page):
        """Extract the full NOC hierarchy from the page"""
        try:
            # The data is in nested <details> tags
            # Get all details elements
            all_details = page.locator('details.nocDetails').all()
            print(f"   Found {len(all_details)} NOC detail sections")
            
            # Also get unit group links (the final level with profiles)
            unit_groups = page.locator('details.nocLI').all()
            print(f"   Found {len(unit_groups)} unit groups")
            
            # Extract data from all details sections
            for detail in all_details:
                try:
                    self._extract_detail_data(detail, page)
                except Exception as e:
                    pass
            
            # Extract unit groups
            for unit_group in unit_groups:
                try:
                    self._extract_unit_group_data(unit_group, page)
                except Exception as e:
                    pass
                    
        except Exception as e:
            print(f"   ❌ Hierarchy extraction error: {str(e)}")
            self.errors.append(f"Hierarchy extraction error: {str(e)}")
    
    def _extract_detail_data(self, detail_element, page):
        """Extract data from a details element (non-unit group)"""
        try:
            # Get the summary element
            summary = detail_element.locator('> summary').first
            if not summary.is_visible():
                return
            
            # Get the ID from the summary (this is the NOC code)
            summary_id_elem = summary.locator('[id]').first
            if summary_id_elem.count() == 0:
                return
                
            noc_code = summary_id_elem.get_attribute('id')
            
            # Get the full title text from the noFontStyle span
            title_element = summary.locator('.noFontStyle').first
            if title_element.count() == 0:
                return
                
            full_text = title_element.inner_text().strip()
            
            # Remove the badge/code number from the beginning
            # The text looks like "10 Specialized middle management..."
            # We want just "Specialized middle management..."
            title = ' '.join(full_text.split()[1:]) if ' ' in full_text else full_text
            
            # Check if already processed
            if any(item['noc_code'] == noc_code for item in self.noc_data):
                return
            
            # Get the first description paragraph if available (direct child only)
            description_element = detail_element.locator('> p').first
            description = ""
            if description_element.count() > 0:
                try:
                    description = description_element.inner_text().strip()
                except:
                    description = f"{title}"
            else:
                description = f"{title}"
            
            # Determine level
            level = self._determine_level(noc_code)
            
            self.noc_data.append({
                'noc_code': noc_code,
                'title': title,
                'description': description,
                'level': level,
                'url': f"https://noc.esdc.gc.ca/Structure/Hierarchy#{noc_code}"
            })
            
            print(f"   ✓ {noc_code}: {title[:50]}...")
            
        except Exception as e:
            pass  # Silent fail for individual items
    
    def _extract_unit_group_data(self, unit_group_element, page):
        """Extract data from a unit group element (final level with profile link)"""
        try:
            # Get the summary element
            summary = unit_group_element.locator('> summary').first
            if not summary.is_visible():
                return
            
            # Get the NOC code from the badge with nocCode class
            badge = summary.locator('.badge.nocCode').first
            if badge.count() == 0:
                return
                
            noc_code = badge.inner_text().strip()
            
            # Get the title from the nocTitle span
            title_element = summary.locator('.nocTitle').first
            if title_element.count() > 0:
                title = title_element.inner_text().strip()
            else:
                # Fallback to extracting from full text
                title_element = summary.locator('.noFontStyle').first
                if title_element.count() == 0:
                    return
                full_text = title_element.inner_text().strip()
                title = ' '.join(full_text.split()[1:]) if ' ' in full_text else full_text
            
            # Check if already processed
            if any(item['noc_code'] == noc_code for item in self.noc_data):
                return
            
            # Get the description paragraph (direct child)
            description_element = unit_group_element.locator('> p').first
            description = ""
            if description_element.count() > 0:
                try:
                    description = description_element.inner_text().strip()
                except:
                    description = title
            else:
                description = title
            
            # Get the profile link
            profile_link = unit_group_element.locator('a[href*="NOCProfile"]').first
            url = "https://noc.esdc.gc.ca/Structure/Hierarchy"
            if profile_link.count() > 0:
                href = profile_link.get_attribute('href')
                url = f"https://noc.esdc.gc.ca{href}" if href and href.startswith('/') else href
            
            # Determine level
            level = self._determine_level(noc_code)
            
            self.noc_data.append({
                'noc_code': noc_code,
                'title': title,
                'description': description,
                'level': level,
                'url': url
            })
            
            print(f"   ✓ {noc_code}: {title[:50]}...")
            
        except Exception as e:
            pass  # Silent fail for individual items
    
    def _extract_all_profiles(self, page):
        """Extract detailed profiles for all NOC entries"""
        total = len(self.noc_data)
        for idx, noc_entry in enumerate(self.noc_data, 1):
            try:
                if idx % 10 == 0 or idx == 1:
                    print(f"   Progress: {idx}/{total} profiles extracted...")
                
                profile_url = noc_entry['url']
                profile_data = self._extract_profile_details(page, profile_url)
                
                # Merge profile data into the NOC entry
                noc_entry.update(profile_data)
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                self.errors.append(f"Profile extraction error for {noc_entry['noc_code']}: {str(e)}")
        
        print(f"   ✅ Completed extracting {total} profiles")
    
    def _extract_profile_details(self, page, profile_url):
        """Extract detailed profile information from a NOC profile page"""
        profile_data = {
            'example_titles': [],
            'index_of_titles': [],
            'main_duties': [],
            'employment_requirements': '',
            'additional_information': '',
            'exclusions': [],
            'broad_category': '',
            'teer': '',
            'major_group': '',
            'sub_major_group': '',
            'minor_group': ''
        }
        
        try:
            # Create a new page for the profile
            profile_page = page.context.new_page()
            profile_page.goto(profile_url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait for content to load
            time.sleep(1)
            
            # Extract Example Titles
            try:
                title_elements = profile_page.locator('h4:has-text("Example titles") ~ ul li, h4:has-text("Example titles") ~ div li').all()
                profile_data['example_titles'] = [elem.inner_text().strip() for elem in title_elements if elem.is_visible()]
            except:
                pass
            
            # Extract Index of Titles (if available)
            try:
                # Look for the "Index of Titles" button/link and click it
                index_button = profile_page.locator('button:has-text("Index of Titles"), a:has-text("Index of Titles"), [data-toggle="collapse"]:has-text("Index of Titles")').first
                if index_button.count() > 0:
                    index_button.click()
                    time.sleep(0.5)  # Wait for expansion
                    
                    # Extract all titles from the expanded section
                    index_elements = profile_page.locator('h4:has-text("Index of Titles") ~ ul li, h4:has-text("Index of Titles") ~ div li, #indexOfTitles li, .index-of-titles li').all()
                    profile_data['index_of_titles'] = [elem.inner_text().strip() for elem in index_elements if elem.is_visible()]
            except:
                pass
            
            # Extract Main Duties
            try:
                duty_elements = profile_page.locator('h4:has-text("Main duties") ~ ul li, h5:has-text("This group performs") ~ ul li').all()
                profile_data['main_duties'] = [elem.inner_text().strip() for elem in duty_elements if elem.is_visible()]
            except:
                pass
            
            # Extract Employment Requirements
            try:
                req_section = profile_page.locator('h4:has-text("Employment requirements")').first
                if req_section.count() > 0:
                    # Get the next sibling elements (could be paragraphs or lists)
                    req_content = []
                    
                    # Try to get list items
                    req_list = profile_page.locator('h4:has-text("Employment requirements") ~ ul li').all()
                    if req_list:
                        req_content = [elem.inner_text().strip() for elem in req_list if elem.is_visible()]
                    else:
                        # Try to get paragraphs
                        req_paras = profile_page.locator('h4:has-text("Employment requirements") ~ p').all()
                        req_content = [elem.inner_text().strip() for elem in req_paras if elem.is_visible()]
                    
                    profile_data['employment_requirements'] = ' | '.join(req_content) if req_content else ''
            except:
                pass
            
            # Extract Additional Information
            try:
                add_info_section = profile_page.locator('h4:has-text("Additional information")').first
                if add_info_section.count() > 0:
                    add_info_list = profile_page.locator('h4:has-text("Additional information") ~ ul li').all()
                    if add_info_list:
                        add_info_content = [elem.inner_text().strip() for elem in add_info_list if elem.is_visible()]
                        profile_data['additional_information'] = ' | '.join(add_info_content)
                    else:
                        add_info_paras = profile_page.locator('h4:has-text("Additional information") ~ p').all()
                        if add_info_paras:
                            add_info_content = [elem.inner_text().strip() for elem in add_info_paras if elem.is_visible()]
                            profile_data['additional_information'] = ' | '.join(add_info_content)
            except:
                pass
            
            # Extract Exclusions
            try:
                exclusion_elements = profile_page.locator('h4:has-text("Exclusions") ~ ul li').all()
                profile_data['exclusions'] = [elem.inner_text().strip() for elem in exclusion_elements if elem.is_visible()]
            except:
                pass
            
            # Extract Breakdown Summary
            try:
                # Broad occupational category
                broad_elem = profile_page.locator('strong:has-text("Broad occupational category") ~ a').first
                if broad_elem.count() > 0:
                    profile_data['broad_category'] = broad_elem.inner_text().strip()
                
                # TEER
                teer_elem = profile_page.locator('strong:has-text("TEER")').first
                if teer_elem.count() > 0:
                    # Get the parent element and extract text after the strong tag
                    parent = teer_elem.locator('..').first
                    if parent.count() > 0:
                        text = parent.inner_text().strip()
                        profile_data['teer'] = text.replace('TEER', '').strip()
                
                # Major group
                major_elem = profile_page.locator('strong:has-text("Major group") ~ a').first
                if major_elem.count() > 0:
                    profile_data['major_group'] = major_elem.inner_text().strip()
                
                # Sub-major group
                sub_major_elem = profile_page.locator('strong:has-text("Sub-major group") ~ a').first
                if sub_major_elem.count() > 0:
                    profile_data['sub_major_group'] = sub_major_elem.inner_text().strip()
                
                # Minor group
                minor_elem = profile_page.locator('strong:has-text("Minor group") ~ a').first
                if minor_elem.count() > 0:
                    profile_data['minor_group'] = minor_elem.inner_text().strip()
            except:
                pass
            
            profile_page.close()
            
        except Exception as e:
            self.errors.append(f"Error extracting profile: {str(e)}")
        
        return profile_data
    
    def _determine_level(self, noc_code):
        """Determine the hierarchy level based on NOC code"""
        # Remove any dots or spaces
        code = noc_code.replace('.', '').replace(' ', '')
        
        if len(code) == 1:
            return "Broad Occupational Category"
        elif len(code) == 2:
            return "Major Group"
        elif len(code) == 3:
            return "Sub-major Group"
        elif len(code) == 4:
            return "Minor Group"
        elif len(code) == 5:
            return "Unit Group"
        else:
            return "Unknown"
    
    def save_to_csv(self, filename='noc_data.csv'):
        """Save scraped data to CSV"""
        if not self.noc_data:
            print("❌ No data to save")
            return
        
        # Convert lists to strings for CSV
        data_for_csv = []
        for item in self.noc_data:
            row = item.copy()
            # Convert list fields to pipe-separated strings
            for key in ['example_titles', 'index_of_titles', 'main_duties', 'exclusions']:
                if key in row and isinstance(row[key], list):
                    row[key] = ' | '.join(row[key])
            data_for_csv.append(row)
        
        df = pd.DataFrame(data_for_csv)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n💾 Data saved to {filename}")
        print(f"   Total records: {len(df)}")
    
    def save_to_json(self, filename='noc_data.json'):
        """Save scraped data to JSON"""
        if not self.noc_data:
            print("❌ No data to save")
            return
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.noc_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Data saved to {filename}")
        print(f"   Total records: {len(self.noc_data)}")
    
    def print_summary(self):
        """Print a summary of the scraped data"""
        if not self.noc_data:
            print("❌ No data scraped")
            return
        
        df = pd.DataFrame(self.noc_data)
        
        print("\n" + "="*60)
        print("📊 SCRAPING SUMMARY")
        print("="*60)
        print(f"Total NOC entries scraped: {len(df)}")
        print("\nBreakdown by level:")
        print(df['level'].value_counts().to_string())
        
        if self.errors:
            print(f"\n⚠️  Errors encountered: {len(self.errors)}")
            print("First 5 errors:")
            for error in self.errors[:5]:
                print(f"   - {error}")


def main():
    """Main function to run the scraper"""
    # Set scrape_profiles=True to extract detailed profile information for each NOC
    scraper = NOCScraper(scrape_profiles=True)
    
    # Run the scraper (set headless=False to see the browser)
    scraper.scrape(headless=True)
    
    # Print summary
    scraper.print_summary()
    
    # Save to both CSV and JSON
    scraper.save_to_csv('noc_data_full.csv')
    scraper.save_to_json('noc_data_full.json')
    
    print("\n✅ Scraping complete!")


if __name__ == "__main__":
    main()
