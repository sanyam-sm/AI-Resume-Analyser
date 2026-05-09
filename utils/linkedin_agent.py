import os
import time
import random
import platform
import threading
import asyncio
import subprocess
import socket
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

try:
    from playwright_stealth import stealth_sync
except ImportError:
    def stealth_sync(context):
        pass


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use (Chrome debugging port)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def find_chrome_executable() -> str:
    """Find Chrome executable path on different platforms."""
    system = platform.system().lower()
    if system == 'windows':
        paths = [
            os.path.expandvars(r'%ProgramFiles%\Google\Chrome\Application\chrome.exe'),
            os.path.expandvars(r'%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe'),
            os.path.expandvars(r'%LocalAppData%\Google\Chrome\Application\chrome.exe'),
        ]
    elif system == 'darwin':
        paths = ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome']
    else:
        paths = ['/usr/bin/google-chrome', '/usr/bin/chromium-browser', '/usr/bin/chromium']
    
    for p in paths:
        if os.path.exists(p):
            return p
    return None


class JobSkipException(Exception):
    pass


class JobApplyAgent:
    CDP_PORT = 9222
    
    def _chrome_user_data_dir(self) -> str:
        """Returns the user's actual Chrome profile directory."""
        system = platform.system().lower()
        if system == 'windows':
            return os.path.expandvars(r'%LocalAppData%\Google\Chrome\User Data')
        elif system == 'darwin':
            return str(Path.home() / 'Library' / 'Application Support' / 'Google' / 'Chrome')
        else:
            return str(Path.home() / '.config' / 'google-chrome')

    def _human_move(self, page):
        x = random.randint(100, 900)
        y = random.randint(80, 650)
        page.mouse.move(x, y, steps=random.randint(5, 20))
        time.sleep(random.uniform(0.1, 0.35))

    def _human_click(self, locator, page):
        self._human_move(page)
        locator.click(timeout=3000)

    def connect_browser(self):
        """Connect to existing Chrome or launch new one with debugging enabled."""
        try:
            loop = asyncio.get_running_loop()
            return self._connect_browser_in_thread()
        except RuntimeError:
            return self._connect_browser_sync()

    def _connect_browser_in_thread(self):
        """Run browser connection in a separate thread to avoid asyncio conflicts."""
        result = {'p': None, 'context': None, 'error': None}
        
        def run_in_thread():
            try:
                p, context = self._connect_browser_sync()
                result['p'] = p
                result['context'] = context
            except Exception as e:
                result['error'] = e
        
        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join(timeout=45)
        
        if result['error']:
            raise result['error']
        if result['p'] is None:
            raise Exception("Browser connection timed out")
        
        return result['p'], result['context']

    def _connect_browser_sync(self):
        """Connect to Chrome via CDP - uses existing login sessions."""
        cdp_url = f"http://127.0.0.1:{self.CDP_PORT}"
        
        # Check if Chrome is already running with debugging enabled
        if is_port_in_use(self.CDP_PORT):
            print(f"[BROWSER] Found Chrome with debugging on port {self.CDP_PORT}")
            try:
                p = sync_playwright().start()
                browser = p.chromium.connect_over_cdp(cdp_url)
                contexts = browser.contexts
                if contexts:
                    context = contexts[0]
                    print("[BROWSER] Connected to existing Chrome session with your login!")
                    return p, context
                else:
                    # Create new context in connected browser
                    context = browser.new_context()
                    return p, context
            except Exception as e:
                print(f"[BROWSER] CDP connection failed: {e}")
        
        # Chrome not running with debugging - need to launch it
        print("[BROWSER] Starting Chrome with debugging enabled...")
        chrome_path = find_chrome_executable()
        if not chrome_path:
            raise Exception("Chrome not found. Please install Google Chrome.")
        
        user_data_dir = self._chrome_user_data_dir()
        
        # Launch Chrome with debugging enabled using user's profile
        cmd = [
            chrome_path,
            f'--remote-debugging-port={self.CDP_PORT}',
            f'--user-data-dir={user_data_dir}',
            '--no-first-run',
            '--no-default-browser-check',
        ]
        
        try:
            # Start Chrome process (don't wait for it)
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[BROWSER] Launched Chrome with your profile")
            
            # Wait for Chrome to start and enable debugging
            for i in range(15):
                time.sleep(1)
                if is_port_in_use(self.CDP_PORT):
                    break
            else:
                raise Exception("Chrome did not start with debugging. Please close all Chrome windows and try again.")
            
            time.sleep(2)  # Extra wait for stability
            
            # Connect via CDP
            p = sync_playwright().start()
            browser = p.chromium.connect_over_cdp(cdp_url)
            contexts = browser.contexts
            if contexts:
                context = contexts[0]
                print("[BROWSER] Connected! Using your existing LinkedIn login.")
                return p, context
            else:
                context = browser.new_context()
                return p, context
                
        except Exception as e:
            error_msg = str(e)
            if "cannot open" in error_msg.lower() or "user data" in error_msg.lower():
                raise Exception(
                    "Chrome profile is locked (another Chrome is running). "
                    "Please close ALL Chrome windows first, then try again. "
                    "Your tabs will reopen automatically when Chrome restarts."
                )
            raise Exception(f"Failed to launch Chrome: {error_msg}")

    def _split_name(self, full_name: str):
        parts = [p for p in (full_name or '').strip().split() if p]
        if not parts:
            return '', ''
        if len(parts) == 1:
            return parts[0], ''
        return parts[0], ' '.join(parts[1:])

    def _fill_common_fields_by_label(self, page, resume_data: dict):
        first_name, last_name = self._split_name(resume_data.get('name', ''))
        email = resume_data.get('email', '')
        phone = resume_data.get('phone', '')
        years = resume_data.get('years_of_experience', '')
        pdf_path = resume_data.get('uploaded_pdf_path', '')

        label_map = [
            (['first name', 'given name'], first_name),
            (['last name', 'surname', 'family name'], last_name),
            (['email'], email),
            (['phone', 'mobile'], phone),
            (['years of experience', 'experience'], years),
        ]

        for labels, value in label_map:
            if not value:
                continue
            for label_text in labels:
                label = page.locator(f"label:has-text('{label_text}')").first
                if label.count() == 0:
                    continue
                field_id = label.get_attribute('for')
                if field_id:
                    inp = page.locator(f"#{field_id}").first
                else:
                    inp = label.locator("xpath=following::input[1]").first
                if inp.count() > 0:
                    inp.fill(str(value))
                    break

        file_inputs = page.locator("input[type='file']")
        if pdf_path and file_inputs.count() > 0:
            file_inputs.first.set_input_files(pdf_path)

    def _unknown_required_exists(self, page) -> bool:
        required = page.locator("input[required], select[required], textarea[required]")
        return required.count() > 0

    def _is_external_or_assessment(self, page) -> bool:
        body = (page.locator("body").inner_text(timeout=2000) or '').lower()
        signals = ['assessment', 'external application', 'continue to apply', 'redirect']
        return any(s in body for s in signals)

    def _click_next_until_submit(self, page, assisted_mode=True):
        for _ in range(10):
            if self._is_external_or_assessment(page):
                raise JobSkipException('assessment_or_external_redirect_detected')

            self._fill_common_fields_by_label(page, self._resume_data)

            submit_btn = page.locator("button:has-text('Submit application'), button:has-text('Submit')").first
            if submit_btn.count() > 0 and submit_btn.is_visible():
                if assisted_mode:
                    return 'filled'
                self._human_click(submit_btn, page)
                return 'applied'

            next_btn = page.locator("button:has-text('Next'), button:has-text('Review')").first
            if next_btn.count() > 0 and next_btn.is_visible():
                self._human_click(next_btn, page)
                time.sleep(random.uniform(0.5, 1.2))
                continue

            if self._unknown_required_exists(page):
                raise JobSkipException('unknown_required_field')
            return 'filled'

        raise JobSkipException('too_many_steps')

    def _handle_linkedin_easy_apply(self, page, assisted_mode=True):
        easy_apply = page.locator("button:has-text('Easy Apply')").first
        if easy_apply.count() == 0:
            raise JobSkipException('easy_apply_not_found')
        self._human_click(easy_apply, page)
        time.sleep(random.uniform(0.8, 1.6))
        return self._click_next_until_submit(page, assisted_mode=assisted_mode)

    def _handle_indeed_apply(self, page, assisted_mode=True):
        easy_apply = page.locator("button:has-text('Apply now'), button:has-text('Easily apply'), a:has-text('Apply now')").first
        if easy_apply.count() == 0:
            raise JobSkipException('easy_apply_not_found')
        self._human_click(easy_apply, page)
        time.sleep(random.uniform(0.8, 1.6))
        return self._click_next_until_submit(page, assisted_mode=assisted_mode)

    def run_assisted(self, jobs: list, resume_data: dict):
        self._resume_data = resume_data
        p, context = self.connect_browser()
        try:
            for job in jobs:
                title = job.get('job_title', 'Unknown role')
                company = job.get('company_name', 'Unknown company')
                url = job.get('apply_url', '')
                source = (job.get('source_platform') or '').lower()

                try:
                    page = context.new_page()
                    page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    yield {'job_title': title, 'company': company, 'status': 'opening', 'message': 'Opened job page'}
                    if 'linkedin' in source:
                        result = self._handle_linkedin_easy_apply(page, assisted_mode=True)
                    elif 'indeed' in source:
                        result = self._handle_indeed_apply(page, assisted_mode=True)
                    else:
                        raise JobSkipException('unsupported_source')
                    yield {'job_title': title, 'company': company, 'status': result, 'message': 'Form filled; stopped before submit'}
                except JobSkipException as e:
                    yield {'job_title': title, 'company': company, 'status': 'skipped', 'reason': str(e)}
                except Exception as e:
                    yield {'job_title': title, 'company': company, 'status': 'failed', 'message': str(e)}
                finally:
                    time.sleep(random.uniform(1.2, 2.4))
        finally:
            context.close()
            p.stop()

    def run_automated(self, jobs: list, resume_data: dict):
        self._resume_data = resume_data
        limited_jobs = jobs[:30]
        p, context = self.connect_browser()
        stealth_sync(context)
        try:
            for job in limited_jobs:
                title = job.get('job_title', 'Unknown role')
                company = job.get('company_name', 'Unknown company')
                url = job.get('apply_url', '')
                source = (job.get('source_platform') or '').lower()

                try:
                    page = context.new_page()
                    page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    yield {'job_title': title, 'company': company, 'status': 'opening', 'message': 'Opened job page'}
                    if 'linkedin' in source:
                        result = self._handle_linkedin_easy_apply(page, assisted_mode=False)
                    elif 'indeed' in source:
                        result = self._handle_indeed_apply(page, assisted_mode=False)
                    else:
                        raise JobSkipException('unsupported_source')
                    yield {'job_title': title, 'company': company, 'status': result, 'message': 'Application submitted'}
                except JobSkipException as e:
                    yield {'job_title': title, 'company': company, 'status': 'skipped', 'reason': str(e)}
                except PlaywrightTimeoutError:
                    yield {'job_title': title, 'company': company, 'status': 'failed', 'message': 'timeout'}
                except Exception as e:
                    yield {'job_title': title, 'company': company, 'status': 'failed', 'message': str(e)}
                finally:
                    time.sleep(random.uniform(1.0, 2.0))
        finally:
            context.close()
            p.stop()
