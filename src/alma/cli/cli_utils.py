from typing import Any, Union
import json


def is_boolean(arg: str) -> bool:
    """Returns true if string represents a boolean value. Else it returns
    false.

    Args:
        arg (str): The string to be assessed.

    Returns:
        bool: True if the string represents a boolean. Else false.
    """
    return arg == "True" or arg == "False"


def is_float(arg: str) -> bool:
    """Check to see if string can be converted to float.

    Args:
        arg (str): The value to be checked.

    Returns:
        bool: True if input string can be converted to float. Else false.
    """
    try:
        float(arg)
    except BaseException:
        return False
    return True


def format_input(dict_in: str) -> dict[str, Any]:
    """Replaces a string representation of dictionary without quotes to a
    dictionary. Note: will not work for lists of dictionaries.

    Args:
        dict_in (str): A string representation of a dictionary without quotes.

    Returns:
        dict[str,str]: The input string in dictionary format with strings for
            both keys and values.
    """
    out = {}
    dict_in = dict_in.strip()
    dict_in = json.loads(dict_in)
    for key in dict_in:
        out[key.strip()] = convert_parameter(dict_in[key])
    return out


def convert_parameter(param: str):
    """Converts the inputed string to the appropriate data type.

    Args:
        param (str): The parameter to be converted.

    Returns:
        Union[bool, int, float, str]: The parameter converted to it's
            appropriate type.
    """
    if isinstance(param, str):
        param = param.strip()
    # Convert dicts
    if isinstance(param, dict) or param[0] == "{":
        return parse_dict(param)
    elif is_boolean(param):
        return bool(param)
    elif param.isdecimal():
        return int(param)
    elif is_float(param):
        return float(param)
    # Convert lists
    elif param[0] == '[':
        param = param[1:-1]
        list_array = param.split(sep=',')
        for idx in range(len(list_array)):
            list_array[idx] = convert_parameter(list_array[idx].strip())
        return list_array
    return param


def parse_dict(dict_in: Union[str, dict[str, str]]) -> dict[str, Any]:
    """Converts a string representation of a dictionary or a dictionary
    containing all strings into a dictionary containing appropriate values.

    Args:
        dict_in (Union[str, dict[str, str]]): Either a string representing a
            dictionary or a dictionary with both keys and values as strings.

    Returns:
        dict[str, Any]: The same dictionary as represented by the string but
            with the values replaced with apropriate types.
    """
    output_dict = {}
    if isinstance(dict_in, str):
        dict_in = json.loads(dict_in)
    for key in dict_in:
        if isinstance(dict_in[key], dict):
            output_dict[key.strip()] = parse_dict(dict_in[key])
        else:
            output_dict[key.strip()] = convert_parameter(dict_in[key])
    return output_dict
