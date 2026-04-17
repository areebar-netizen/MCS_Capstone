from django import template

register = template.Library()

@register.filter
def get_saved_value(data_dict, key):
    """Get value from dictionary by key"""
    try:
        return data_dict.get(key, None)
    except (AttributeError, TypeError):
        return None

@register.filter
def get_saved_value_with_default(data_dict, key):
    """Get value from dictionary by key with default - expects format: data_dict|get_saved_value_with_default:key:default"""
    # This will be called with the default value as part of the filter chain
    return data_dict.get(key, '')
