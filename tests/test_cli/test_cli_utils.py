import unittest
from alma.cli.cli_utils import (
    is_boolean,
    is_float,
    format_input,
    convert_parameter,
    parse_dict
)


class TestCliUtils(unittest.TestCase):

    def test_is_boolean(self):
        """Tests that is_boolean can correclty identify a boolean value."""
        self.assertTrue(is_boolean("True"))
        self.assertTrue(is_boolean("False"))
        self.assertFalse(is_boolean("Ture"))
        self.assertFalse(is_boolean("Flase"))

    def test_if_float(self):
        """Tests that is_float can correclty identify a value that can be
        cast as a float.
        """
        self.assertTrue(is_float("1.5"))
        self.assertFalse(is_float("a"))

    def test_parse_dict(self):
        """Tests that the parse_dict function parses dictionaries correctly."""
        test_dict = ' { "  key1"  :  "  1", "key2  ":  "2    "   } '
        self.assertCountEqual(parse_dict(test_dict),
                              {"key1":  "1", "key2":  "2"},
                              msg="The parse_dict function can't pares "
                                  "strings correctly.")
        test_dict = ' { "  key1"  :  "  1", "key2  ":  {"  key3"  :  "  3",' \
                    ' "  key4"  :  "  4"}   } '
        self.assertCountEqual(
            parse_dict(test_dict),
            {"key1":  "1", "key2":  {"key3":  "3", "key4":  "4"}},
            msg="The parse_dict function can't parse nested dictionaries "
                "correctly."
        )

    def test_convert_parameter(self):
        """Tests that the convert_parameter function is able to convert string
        representations of parameters correctly.
        """
        self.assertEqual(convert_parameter("a string"),
                         "a string",
                         msg="convert_parameter function fails to convert "
                             "strings properly.")
        self.assertEqual(convert_parameter("1"),
                         int(1),
                         msg="convert_parameter function fails to convert "
                             "integers properly.")
        self.assertEqual(convert_parameter("1.5"),
                         float(1.5),
                         msg="convert_parameter function fails to convert "
                             "floats properly.")
        test_dict = ' { "  key1"  :  "  1", "key2  ":  "2    "   } '
        self.assertCountEqual(convert_parameter(test_dict),
                              {"key1":  "1", "key2":  "2"},
                              msg="convert_parameter function fails to "
                              "convert dictionaries properly.")
        test_list = ' [  1 ,  2  , 3   ]  '
        self.assertCountEqual(convert_parameter(test_list),
                              [1, 2, 3],
                              msg="convert_parameter function fails to"
                                  "convert lists properly.")

    def test_format_input(self):
        """Tests that the format_input function formats string representations
        of dictionaries correctly.
        """
        test_dict = '{"key1  "  :  "1 "  , " key2": " 1.5", "dict_key"  : ' \
                    '{"key3" : "v3" , "key4" : "[1,2,3]"}, "  key5 "  : "' \
                    '  True"}'
        target_dict = {"key1": 1,
                       "key2": 1.5,
                       "dict_key": {"key3": "v3",
                                    "key4": [1, 2, 3]},
                       "key5": True}
        self.assertCountEqual(format_input(test_dict),
                              target_dict)
