from django.shortcuts import render

# Create your views here.
<<<<<<< Updated upstream
=======

MODEL_SERVICE = PredictionService(models_dir=Path(settings.BASE_DIR.parent)/ 'core_engine' / 'artifacts', model_name='xgboost')

def email_entry(request):
    """Landing page for email entry"""
    return render(request, 'email_entry.html')

def send_otp(request):
    """Generate and send OTP for email verification"""
    if request.method == 'POST':
        email = request.POST.get('email')
        
        if not email:
            return JsonResponse({'error': 'Email is required'}, status=400)
        
        # Generate 6-digit OTP
        otp_code = ''.join(random.choices(string.digits, k=6))
        print(f"DEBUG: OTP for {email} is {otp_code}")
        
        # Save OTP to database
        from .models import EmailOTP
        EmailOTP.objects.filter(email=email).delete()  # Remove any existing OTPs
        EmailOTP.objects.create(email=email, otp_code=otp_code)
        
        # Send OTP email
        try:
            from django.core.mail import send_mail
            subject = 'BrainWave - Your OTP Code'
            message = f'Your BrainWave verification code is: {otp_code}\n\nThis code will expire in 5 minutes.'
            from_email = 'noreply@brainwave.com'
            recipient_list = [email]
            
            send_mail(
                subject,
                message,
                from_email,
                recipient_list,
                fail_silently=False,
            )
            print(f"OTP email sent successfully to {email}")
        except Exception as e:
            print(f"Failed to send OTP email: {e}")
            # For development, you might want to still show the OTP in console
            print(f"DEBUG: OTP for {email} is {otp_code}")
            # In production, you might want to handle this differently
            # For now, we'll continue with the flow even if email fails
        
        # Redirect to OTP verification page
        return render(request, 'otp_verification.html', {'email': email})
    
    return redirect('/')

def verify_otp(request):
    """Verify OTP and handle user routing"""
    if request.method == 'POST':
        email = request.POST.get('email')
        otp_code = request.POST.get('otp_code')
        
        if not email or not otp_code:
            return JsonResponse({'error': 'Email and OTP are required'}, status=400)
        
        from .models import EmailOTP, UserProfile
        
        # Verify OTP
        try:
            otp_record = EmailOTP.objects.get(email=email, otp_code=otp_code)
            if not otp_record.is_valid():
                return JsonResponse({'error': 'OTP has expired'}, status=400)
        except EmailOTP.DoesNotExist:
            return JsonResponse({'error': 'Invalid OTP'}, status=400)
        
        # OTP is valid - set session
        request.session['user_email'] = email
        request.session.modified = True
        
        # Clean up OTP
        otp_record.delete()
        
        # Check if user exists and has completed survey
        user_exists = UserProfile.objects.filter(email=email).exists()
        if user_exists:
            user_profile = UserProfile.objects.get(email=email)
            # Check if profile is complete (has basic info)
            if user_profile.name and user_profile.academic_level:
                return redirect('/dashboard/')
            else:
                return redirect('/onboarding/')
        else:
            # New user - redirect to survey
            return redirect('/onboarding/')
    
    return redirect('/')

def dashboard_view(request):
    """Dashboard view with user profile and focus tracking data"""
    # Security check - ensure user is authenticated via OTP
    user_email = request.session.get('user_email')
    if not user_email:
        return redirect('/')
    
    # Get user profile data from database
    from .models import UserProfile
    try:
        user_profile = UserProfile.objects.get(email=user_email)
    except UserProfile.DoesNotExist:
        return redirect('/onboarding/')
    
    # Import data service
    from .services.data_service import FocusDataService
    data_service = FocusDataService(user_email)
    
    # Helper functions to map indices to human-readable text
    def get_academic_level_text(level_id):
        levels = {
            0: 'High School',
            1: 'Undergraduate', 
            2: 'Graduate/Masters',
            3: 'PhD/Doctoral',
            4: 'Professional/Continuing Education',
            5: 'Other'
        }
        return levels.get(int(level_id), 'Not Set')
    
    def get_sleep_quality_text(quality_id):
        qualities = {
            0: 'Poor (frequently disrupted)',
            1: 'Fair (occasional issues)',
            2: 'Good (generally restful)',
            3: 'Excellent (consistently deep)'
        }
        return qualities.get(int(quality_id), 'Not Set')
    
    def get_alert_time_text(time_id):
        times = {
            0: 'Early Morning (5am-9am)',
            1: 'Mid-Morning to Afternoon (9am-5pm)',
            2: 'Evening to Late Night (5pm-12am)',
            3: 'Late Night (12am-5am)'
        }
        return times.get(int(time_id), 'Not Set')
    
    def get_learning_style_text(style_id):
        styles = {
            0: 'Visual Learner',
            1: 'Auditory Learner',
            2: 'Kinesthetic Learner',
            3: 'Reading/Writing Learner'
        }
        return styles.get(int(style_id), 'Not Set')
    
    def get_session_length_text(length_id):
        lengths = {
            0: 'Short (15-30 min)',
            1: 'Medium (30-60 min)',
            2: 'Long (60-90 min)',
            3: 'Extended (90+ min)'
        }
        return lengths.get(int(length_id), 'Not Set')
    
    def get_sound_environment_text(env_id):
        environments = {
            0: 'Complete Silence',
            1: 'White Noise',
            2: 'Soft Music',
            3: 'Nature Sounds',
            4: 'Cafe/Background Noise',
            5: 'Instrumental Music'
        }
        return environments.get(int(env_id), 'Not Set')
    
    def get_study_time_text(time_id):
        times = {
            0: 'Early Morning',
            1: 'Morning',
            2: 'Afternoon', 
            3: 'Evening',
            4: 'Night'
        }
        return times.get(int(time_id), 'Not Set')
    
    # Create user profile snapshot with human-readable text
    profile_snapshot = {
        'name': user_profile.name,
        'age': user_profile.age,
        'academic_level': get_academic_level_text(user_profile.academic_level),
        'sleep_hours': user_profile.sleep_hours,
        'sleep_quality': get_sleep_quality_text(user_profile.sleep_quality),
        'learning_style': get_learning_style_text(user_profile.learning_style),
        'caffeine_servings': user_profile.caffeine_servings,
        'procrastination_level': user_profile.procrastination_level,
        'main_goals': user_profile.main_goals.strip('[]').replace("'", '').split(',')[0] if user_profile.main_goals else 'Not Set',
        'sound_environment': get_sound_environment_text(user_profile.sound_environment),
        'study_location': user_profile.study_location.strip('[]').replace("'", '').split(',')[0] if user_profile.study_location else 'Not Set',
        'session_length': get_session_length_text(user_profile.session_length),
        'study_time_of_day': get_study_time_text(user_profile.study_time_of_day),
        'alert_time': get_alert_time_text(user_profile.alert_time)
    }
    
    # Get focus tracking data
    recent_sessions = data_service.get_recent_sessions()
    session_stats = data_service.get_session_average_stats()
    brainwave_data = data_service.get_brainwave_averages()
    recommendations = data_service.get_recommendations()
    calendar_data = data_service.get_calendar_data()
    
    # Mock data for demonstration
    context = {
        'user': request.user if request.user.is_authenticated else None,
        'user_profile': profile_snapshot,
        'recent_sessions': recent_sessions,
        'session_stats': session_stats,
        'brainwave_data': brainwave_data,
        'recommendations': recommendations,
        'calendar_data': calendar_data,
        'current_month': calendar_data[0]['day'] if calendar_data else 1,
        'current_year': timezone.now().year if calendar_data else 2026
    }
    
    return render(request, 'dashboard.html', context)

def onboarding_view(request):
    """Render the onboarding page with multi-step form handling"""
    
    # Security check - ensure user is authenticated via OTP
    user_email = request.session.get('user_email')
    if not user_email:
        return redirect('/')
    
    # Define all sections and their questions
    SECTIONS = {
        1: {
            'title': 'SECTION 1: BASIC INFORMATION',
            'questions': [
                {'id': 'name', 'type': 'text', 'label': "What's your name?", 'placeholder': 'Enter your name'},
                {'id': 'age', 'type': 'number', 'label': 'How old are you?', 'placeholder': 'Enter your age', 'min': 13, 'max': 99},
                {'id': 'academic_level', 'type': 'radio', 'label': 'What\'s your academic level?', 
                 'options': ['High School', 'Undergraduate', 'Graduate/Masters', 'PhD/Doctoral', 'Professional/Continuing Education', 'Other']}
            ]
        },
        2: {
            'title': 'SECTION 2: YOUR NATURAL RHYTHMS',
            'questions': [
                {'id': 'alert_time', 'type': 'radio', 'label': 'When do you feel most alert and energetic?',
                 'options': ['Early Morning (5am-9am) - Morning Person', 'Mid-Morning to Afternoon (9am-5pm) - Intermediate', 
                          'Evening to Late Night (5pm-12am) - Night Owl', 'Late Night (12am-5am) - Night Owl']},
                {'id': 'sleep_hours', 'type': 'range', 'label': 'How many hours of sleep do you typically get per night?', 'min': 4, 'max': 11, 'default': 7},
                {'id': 'sleep_quality', 'type': 'radio', 'label': 'How would you rate your sleep quality?',
                 'options': ['Poor (frequently disrupted, unrefreshing)', 'Fair (occasional issues)', 
                          'Good (generally restful)', 'Excellent (consistently deep and refreshing)']}
            ]
        },
        3: {
            'title': 'SECTION 3: CAFFEINE & STIMULANTS',
            'questions': [
                {'id': 'consumes_caffeine', 'type': 'radio', 'label': 'Do you consume caffeine?', 'options': ['Yes', 'No']},
                {'id': 'caffeine_types', 'type': 'checkbox', 'label': 'What type(s) of caffeine? (if Yes)',
                 'options': ['Coffee', 'Tea (black/green)', 'Energy drinks', 'Soda', 'Other']},
                {'id': 'caffeine_servings', 'type': 'number', 'label': 'How many servings per day? (if Yes)', 'min': 1, 'max': 10},
                {'id': 'caffeine_timing', 'type': 'radio', 'label': 'When do you typically consume caffeine? (if Yes)',
                 'options': ['Early morning only (before 10am)', 'Morning to noon (before 12pm)', 
                          'Throughout the day (morning to afternoon)', 'Anytime (including evenings)']}
            ]
        },
        4: {
            'title': 'SECTION 4: LEARNING STYLE & PREFERENCES',
            'questions': [
                {'id': 'learning_style', 'type': 'radio', 'label': 'How do you learn best?',
                 'options': ['Visual (diagrams, charts, videos, reading)', 'Auditory (listening to lectures, discussions, audio)', 
                          'Kinesthetic (hands-on practice, movement, doing)', 'Reading/Writing (taking notes, writing summaries)', 'Not sure']},
                {'id': 'study_subjects', 'type': 'checkbox', 'label': 'What subjects/topics do you typically study?',
                 'options': ['Math/Statistics/Calculus', 'Sciences (Biology, Chemistry, Physics)', 'Reading/Literature/Humanities', 
                          'Writing/Essays/Creative Work', 'Languages', 'Programming/Computer Science', 'Memorization-heavy subjects', 'Other']}
            ]
        },
        5: {
            'title': 'SECTION 5: CURRENT STUDY HABITS',
            'questions': [
                {'id': 'session_length', 'type': 'radio', 'label': 'How long do you typically study in one session?',
                 'options': ['15-30 minutes', '30-45 minutes', '45-60 minutes', '1-2 hours', '2+ hours', 'It varies widely']},
                {'id': 'takes_breaks', 'type': 'radio', 'label': 'Do you currently take breaks during study sessions?',
                 'options': ['No, I study straight through', 'Yes, when I feel tired', 'Yes, every 25-30 minutes (Pomodoro-style)', 
                          'Yes, every 45-60 minutes', 'Other']},
                {'id': 'study_time_of_day', 'type': 'radio', 'label': 'What time of day do you usually study?',
                 'options': ['Early Morning (5am-9am)', 'Mid-Morning (9am-12pm)', 'Afternoon (12pm-5pm)', 
                          'Evening (5pm-9pm)', 'Night (9pm-12am)', 'Late Night (12am-5am)', 'Varies by day']},
                {'id': 'procrastination_level', 'type': 'range', 'label': 'On a scale of 1-10, how much do you struggle with procrastination?', 'min': 1, 'max': 10, 'default': 5}
            ]
        },
        6: {
            'title': 'SECTION 6: STUDY ENVIRONMENT PREFERENCES',
            'questions': [
                {'id': 'study_location', 'type': 'checkbox', 'label': 'Where do you typically study?',
                 'options': ['Library or study hall', 'Home (quiet room/desk)', 'Home (shared/noisy space)', 
                          'Coffee shop or café', 'Outdoors', 'Other']},
                {'id': 'sound_environment', 'type': 'radio', 'label': 'What sound environment do you prefer while studying?',
                 'options': ['Complete silence', 'White noise or brown noise', 'Nature sounds or ambient sounds', 
                          'Lo-fi or instrumental music', 'Classical music', 'Music with lyrics', 'Coffee shop ambience', 
                          'I\'m not sure yet / want to experiment', 'Other']},
                {'id': 'lighting_preference', 'type': 'radio', 'label': 'What lighting do you prefer for studying?',
                 'options': ['Natural daylight', 'Bright artificial light', 'Dim or warm lighting', 
                          'Mix of natural and artificial', 'Not sure / no strong preference']}
            ]
        },
        7: {
            'title': 'SECTION 7: DISTRACTIONS & FOCUS',
            'questions': [
                {'id': 'phone_location', 'type': 'radio', 'label': 'Where is your phone when you study?',
                 'options': ['Right next to me on my desk', 'In my pocket or bag (nearby)', 'In another room', 
                          'On "Do Not Disturb" mode nearby', 'I don\'t have specific habits around this']},
                {'id': 'distractions', 'type': 'checkbox', 'label': 'Which of these distractions affect you most?',
                 'options': ['Social media notifications', 'Text messages', 'Background noise/people talking', 
                          'Hunger/thirst', 'Uncomfortable seating', 'Temperature (too hot/cold)', 
                          'Wandering thoughts/daydreaming', 'Other']}
            ]
        },
        8: {
            'title': 'SECTION 8: PHYSICAL ACTIVITY & LIFESTYLE',
            'questions': [
                {'id': 'exercise_frequency', 'type': 'radio', 'label': 'How often do you engage in physical activity or exercise?',
                 'options': ['Daily', '3-5 times per week', '1-2 times per week', 'Rarely', 'Never']},
                {'id': 'eating_timing', 'type': 'radio', 'label': 'How long after eating do you typically study?',
                 'options': ['Immediately or within 30 minutes', '30 minutes to 1 hour', '1-2 hours', 
                          '2-4 hours', '4+ hours or on empty stomach', 'It varies']}
            ]
        },
        9: {
            'title': 'SECTION 9: HEALTH & WELLNESS (OPTIONAL)',
            'questions': [
                {'id': 'health_conditions', 'type': 'checkbox', 'label': 'Do you have any conditions that might affect your focus? (Optional)',
                 'options': ['ADHD or attention difficulties', 'Anxiety or high stress', 'Depression', 
                          'Sleep disorder', 'None', 'Prefer not to say', 'Other']}
            ]
        },
        10: {
            'title': 'SECTION 10: GOALS & EXPECTATIONS',
            'questions': [
                {'id': 'main_goals', 'type': 'checkbox', 'label': 'What are your main goals with this app?',
                 'options': ['Improve focus and concentration', 'Study more efficiently (less time, better results)', 
                          'Identify my optimal study environment', 'Build better study habits', 'Reduce procrastination', 
                          'Track my progress over time', 'Understand when I\'m most productive', 'Other']},
                {'id': 'study_effectiveness', 'type': 'range', 'label': 'How would you rate your current study effectiveness?', 'min': 1, 'max': 10, 'default': 5}
            ]
        }
    }
    
    # Handle form submission
    if request.method == 'POST':
        # Get current step from form data or default to 1
        current_step = int(request.POST.get('current_step', 1))
        
        # Store form data in session
        if 'onboarding_data' not in request.session:
            request.session['onboarding_data'] = {}
        
        # Save current step data
        step_data = {}
        section = SECTIONS.get(current_step, {})
        for question in section.get('questions', []):
            field_name = question['id']
            if question['type'] == 'checkbox':
                step_data[field_name] = request.POST.getlist(field_name)
            else:
                step_data[field_name] = request.POST.get(field_name, '')
        
        # Update session data
        request.session['onboarding_data'].update(step_data)
        request.session.modified = True
        
        # Move to next step
        next_step = current_step + 1
        
        if next_step <= 10:
            # Redirect to next step
            return redirect(f'/onboarding/?step={next_step}')
        else:
            # Complete onboarding - save to database and redirect to dashboard
            try:
                from .models import UserProfile
                
                # Get user email from session
                user_email = request.session.get('user_email')
                if not user_email:
                    return redirect('/')
                
                # Get all onboarding data
                onboarding_data = request.session.get('onboarding_data', {})
                
                # Handle multi-select fields (checkboxes)
                caffeine_types_list = onboarding_data.get('caffeine_types', [])
                study_subjects_list = onboarding_data.get('study_subjects', [])
                distractions_list = onboarding_data.get('distractions', [])
                
                # Check if user consumes caffeine
                consumes_caffeine = onboarding_data.get('consumes_caffeine', 'false').lower() == 'true'
                
                # If user doesn't consume caffeine, set caffeine details to None/empty defaults
                if not consumes_caffeine:
                    caffeine_types_list = None  # Explicitly set to None
                    caffeine_servings = 0
                    caffeine_timing = ''
                else:
                    # Only process caffeine data if user actually consumes caffeine
                    caffeine_types_list = onboarding_data.get('caffeine_types', [])
                    caffeine_servings = int(onboarding_data.get('caffeine_servings', 0))
                    caffeine_timing = onboarding_data.get('caffeine_timing', '')
                
                profile_data = {
                    'email': user_email,  # Use 'email' to match model field name
                    'name': onboarding_data.get('name', ''),
                    'age': int(onboarding_data.get('age', 0)),
                    'academic_level': onboarding_data.get('academic_level', ''),
                    
                    # Section 2: Rhythms
                    'alert_time': onboarding_data.get('alert_time', ''),
                    'sleep_hours': float(onboarding_data.get('sleep_hours', 7)),
                    'sleep_quality': onboarding_data.get('sleep_quality', ''),
                    
                    # Section 3: Caffeine
                    'consumes_caffeine': consumes_caffeine,
                    'caffeine_types': ', '.join(caffeine_types_list) if caffeine_types_list and isinstance(caffeine_types_list, list) else '',
                    'caffeine_servings': caffeine_servings,
                    'caffeine_timing': caffeine_timing,
                    
                    # Section 4: Styles
                    'learning_style': onboarding_data.get('learning_style', ''),
                    'study_subjects': ', '.join(study_subjects_list) if isinstance(study_subjects_list, list) else str(study_subjects_list),
                    
                    # Section 5: Habits
                    'session_length': onboarding_data.get('session_length', ''),
                    'takes_breaks': onboarding_data.get('takes_breaks', ''),
                    'study_time_of_day': onboarding_data.get('study_time_of_day', ''),
                    'procrastination_level': int(onboarding_data.get('procrastination_level', 0)),
                    
                    # Section 6 & 7: Environment & Distractions
                    'study_location': onboarding_data.get('study_location', ''),
                    'sound_environment': onboarding_data.get('sound_environment', ''),
                    'lighting_preference': onboarding_data.get('lighting_preference', ''),
                    'phone_location': onboarding_data.get('phone_location', ''),
                    'distractions': ', '.join(distractions_list) if isinstance(distractions_list, list) else str(distractions_list),
                    
                    # Section 8 & 9: Lifestyle & Health
                    'exercise_frequency': onboarding_data.get('exercise_frequency', ''),
                    'eating_timing': onboarding_data.get('eating_timing', ''),
                    'health_conditions': onboarding_data.get('health_conditions', ''),
                    
                    # Section 10: Goals
                    'main_goals': onboarding_data.get('main_goals', ''),
                    'study_effectiveness': int(onboarding_data.get('study_effectiveness', 0))
                }
                
                # Save or update UserProfile in database
                profile, created = UserProfile.objects.update_or_create(
                    email=user_email,  # Use 'email' to match model field name
                    defaults=profile_data
                )
                
                # Clear session data
                if 'onboarding_data' in request.session:
                    del request.session['onboarding_data']
                request.session.modified = True
                
                print(f"User profile {'created' if created else 'updated'} for {user_email}")
                
            except Exception as e:
                print(f"Error saving user profile: {e}")
                # Continue to dashboard even if save fails
            
            return redirect('/dashboard/')
    
    # Handle GET request
    current_step = int(request.GET.get('step', 1))
    progress_percentage = (current_step / 10) * 100
    
    # Get saved data from session
    onboarding_data = request.session.get('onboarding_data', {})
    
    # Get current section data
    current_section = SECTIONS.get(current_step, SECTIONS[1])
    
    # Add saved data to context for template access
    context = {
        'current_step': current_step,
        'progress_percentage': int(progress_percentage),
        'section': current_section,
        'user': request.user if request.user.is_authenticated else None,
        'session_history': [],  # Would be populated from database
        'focus_percentage': None,  # Would be calculated from current session
        'saved_data': onboarding_data,
    }
    
    return render(request, 'onboarding.html', context)

@csrf_exempt
def prediction_view(request):
    """Generates the live preditictions for EEG data streamed from the device"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=400)
    #Live prediction integration
    try:
        data = json.loads(request.body)
        rows = data.get('rows', [])

        if not rows:
            return JsonResponse({'error': 'No rows provided'}, status = 400)

        user_profile = UserProfile.objects.get(email=user_email)

        result = MODEL_SERVICE.run(rows)
        if result.get('ok') == False:
            return JsonResponse(result, status=400)
        
    except Exception as e:
        print(f'Error in prediction view: {e}')
        return JsonResponse({'error': str(e)}, status = 500)

@csrf_exempt
def start_realtime_eeg_view(request):
    """Start real-time EEG inference with per-second streaming"""
    print("Starting Real-time EEG Session")
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=400)
    
    try:
        # Get duration from request
        data = json.loads(request.body) if request.body else {}
        duration = int(data.get('duration', 1))
        
        # Trigger real-time Celery task
        task = run_live_inference_streaming.delay(user_email, duration)
        
        # Store task ID in session
        request.session['current_eeg_task_id'] = task.id
        request.session['realtime_session_active'] = True
        request.session.modified = True

        
        return JsonResponse({
            'ok': True,
            'message': 'Real-time EEG inference started',
            'task_id': task.id,
            'duration_minutes': duration,
            'session_type': 'realtime',
            'status': 'initializing'
        })

        
        
    except Exception as e:
        print(f'Error starting real-time EEG task: {e}')
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def stop_realtime_eeg_view(request):
    """Stop real-time EEG session and get final summary"""
    print("🛑 Stopping Real-time EEG Session")
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    
    try:
        # Clear realtime session flag
        request.session['realtime_session_active'] = False
        request.session.modified = True
        
        # Get task ID from session
        task_id = request.session.get('current_eeg_task_id')
        if not task_id:
            return JsonResponse({'error': 'No active EEG task found'}, status=400)
        
        # Check task status
        task_result = get_task_status(task_id)
        
        if task_result['status'] == 'SUCCESS':
            result = task_result['result']
            return JsonResponse({
                'ok': True,
                'status': 'completed',
                'session_id': result.get('session_id'),
                'final_summary': result.get('final_summary'),
                'csv_file_path': result.get('csv_file_path'),
                'duration_minutes': result.get('duration_minutes')
            })
        elif task_result['status'] == 'FAILURE':
            return JsonResponse({
                'ok': False,
                'status': 'error',
                'message': 'EEG inference task failed',
                'error': str(task_result.get('result', 'Unknown error'))
            })
        else:
            return JsonResponse({
                'ok': True,
                'status': task_result['status'],
                'message': 'EEG inference still in progress',
                'task_id': task_id
            })
            
    except Exception as e:
        print(f'Error stopping real-time EEG task: {e}')
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def get_realtime_eeg_status_view(request):
    """Get real-time status and current focus data"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    
    try:
        # Get task ID from session
        task_id = request.session.get('current_eeg_task_id')
        if not task_id:
            return JsonResponse({'error': 'No active EEG task found'}, status=400)
        
        # Check if realtime session is active
        is_active = request.session.get('realtime_session_active', False)
        
        # Get task status
        task_result = get_task_status(task_id)
        
        return JsonResponse({
            'ok': True,
            'task_id': task_id,
            'status': task_result['status'],
            'is_realtime_active': is_active,
            'result': task_result.get('result') if task_result['status'] == 'SUCCESS' else None
        })
        
    except Exception as e:
        print(f'Error getting real-time EEG status: {e}')
        return JsonResponse({'error': str(e)}, status=500)

EEGSERVICE = EEGService()  
@csrf_exempt
def start_live_eeg_view(request):
    """Start EEG inference as a Celery task"""
    print("Starting EEG inference task")
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=400)
    
    task_id = request.session.get('current_eeg_task_id')
    #cache.set(f"stop eeg task{task_id}", False, timeout=60*60)
   

    try:
        # Get duration from request (default to 1 minute)
        data = json.loads(request.body) if request.body else {}
        duration = int(data.get('duration', 1))
        
        # Trigger Celery task
        task = run_live_inference.delay(user_email, duration)
        
        # Store task ID in session for status checking
        request.session['current_eeg_task_id'] = task.id
        request.session.modified = True
        
        return JsonResponse({
            'ok': True,
            'message': 'EEG inference task started',
            'task_id': task.id,
            'duration_minutes': duration,
            'status': 'processing'
        })
        
    except Exception as e:
        print(f'Error starting EEG task: {e}')
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def stop_live_eeg_view(request):
    """Check EEG inference task status and return results"""
    print("Checking EEG task status")
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    
    try:
        # Get task ID from session
        task_id = request.session.get('current_eeg_task_id')
        if not task_id:
            return JsonResponse({'error': 'No active EEG task found'}, status=400)
        
        
        # Check task status
        task_result = get_task_status(task_id)
        
        if task_result['status'] == 'SUCCESS':
            # Task completed successfully
            result = task_result['result']
            return JsonResponse({
                'ok': True,
                'status': 'completed',
                'session_id': result.get('session_id'),
                'final_result': result.get('final_result'),
                'raw_data_path': result.get('raw_data_path'),
                'duration_minutes': result.get('duration_minutes')
            })
        elif task_result['status'] == 'PENDING':
            # Task still running
            result = task_result['result']
            #cache.set(f"stop eeg task{task_id}", True, timeout=60*60)  # Reset stop flag for next check
            return JsonResponse({
                'ok': True,
                'status': 'processing',
                'message': 'EEG inference still in progress',
                'task_id': task_id,
            })
        elif task_result['status'] == 'FAILURE':
            # Task failed
            return JsonResponse({
                'ok': False,
                'status': 'error',
                'message': 'EEG inference task failed',
                'error': str(task_result.get('result', 'Unknown error'))
            })
        else:
            return JsonResponse({
                'ok': True,
                'status': task_result['status'],
                'task_id': task_id
            })
            
    except Exception as e:
        print(f'Error checking EEG task status: {e}')
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def eeg_task_status_view(request):
    """Check the status of an ongoing EEG inference task"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=401)
    
    try:
        # Get task ID from session or request parameter
        task_id = request.GET.get('task_id') or request.session.get('current_eeg_task_id')
        if not task_id:
            return JsonResponse({'error': 'No task ID provided'}, status=400)
        
        # Check task status
        task_result = get_task_status(task_id)
        
        return JsonResponse({
            'ok': True,
            'task_id': task_id,
            'status': task_result['status'],
            'result': task_result['result'] if task_result['status'] == 'SUCCESS' else None
        })
        
    except Exception as e:
        print(f'Error checking task status: {e}')
        return JsonResponse({'error': str(e)}, status=500)

    

def upload_csv_view(request):
    """Upload csv online and recvieve predictions for it"""

    if request.method == 'POST' and request.FILES.get('csv_file'):
        file = request.FILES['csv_file']
        try:
            rows = []
            dfile = file.read().decode('utf-8').splitlines()
            reader = csv.reader(dfile)
            next(reader)

            for row in reader:
                rows.append([float(x) for x in row])
            result = MODEL_SERVICE.run(rows)

            return JsonResponse(result)
        
        except Exception as e:
            print(f'Error processsing uploaded csv: {e}')
            return JsonResponse({'error': str(e)}, status=500)
    return render(request, 'upload_csv.html')

def test_csv():
    print("Running test_csv...")  # add this

    rows = []
    with open(r"C:\Users\binom\OneDrive\Desktop\KeystoneProject\MCS_Capstone\dataset\our_data\areeba_new\areeba_concentrating_3min.csv") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append([float(x) for x in row])

    result = MODEL_SERVICE.run(rows)
    print("RESULT:", result)

def recommendation_view(request):
    """Generate Recommendations based on user profile and focus data for each session"""
    user_email = request.session.get('user_email')
    if not user_email:
        return JsonResponse({'error': 'Unauthorized'}, status=400)
    
    return JsonResponse({'message': 'Recommendations feature coming soon'})

def end_session(request):
    """End the current EEG session by creating a stop signal file"""
    if request.method == 'POST':
        user_email = request.session.get('user_email')
        if not user_email:
            return JsonResponse({'error': 'Unauthorized'}, status=400)
        
        try:
            # Create stop signal file for the user
            import os
            stop_signal_path = f"/tmp/stop_session_{user_email}.flag"
            
            # Create the stop signal file
            with open(stop_signal_path, 'w') as f:
                f.write('stop')
            
            return JsonResponse({
                'success': True,
                'message': 'Session termination signal sent'
            })
            
        except Exception as e:
            print(f'Error creating stop signal: {e}')
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Method not allowed'}, status=405)
    



>>>>>>> Stashed changes
