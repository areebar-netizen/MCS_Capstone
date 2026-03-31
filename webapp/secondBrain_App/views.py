from django.shortcuts import render, redirect
from django.http import JsonResponse

# Create your views here.

def dashboard_view(request):
    """Dashboard view with user profile and focus tracking data"""
    # Get user profile data from session or database
    user_profile = {}
    if 'user_profile' in request.session:
        user_profile = request.session['user_profile']
    
    # Mock data for demonstration
    context = {
        'user': request.user,
        'user_profile': user_profile,
        'recent_sessions': [
            {
                'name': 'Morning Study Session',
                'time': 'Today, 9:30 AM',
                'duration': '45 min',
                'focus': 8.2,
                'states': {'concentrated': 65, 'neutral': 25, 'relaxed': 10}
            },
            {
                'name': 'Afternoon Review',
                'time': 'Today, 2:15 PM',
                'duration': '30 min',
                'focus': 6.8,
                'states': {'concentrated': 45, 'neutral': 40, 'relaxed': 15}
            }
        ]
    }
    
    return render(request, 'dashboard.html', context)

def onboarding_view(request):
    """Render the onboarding page with multi-step form handling"""
    
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
            # Complete onboarding - redirect to dashboard
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
