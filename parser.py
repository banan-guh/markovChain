INVISIBLE = {0x034F, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF, 0x00AD, }


def parse_flags(string, flags, flags_input_float, is_last = False, last_num = 1):
    for char in string:
        if ord(char) in INVISIBLE:
            string = string.replace(char, '')
    
    params = string.split()
    
    key_value_flags = dict.fromkeys(flags, False)
    input_key_value_flags = dict.fromkeys(flags_input_float, -1.0)

    for flag in flags:
        if flag in params:
            key_value_flags[flag] = True
            params.remove(flag)


    for param in params[:]: # makes a new object so I don't iterate while modifying
        for flag in flags_input_float:
            if param.startswith(flag):
                string_length = len(flag)
                trunc_string = param[string_length:]
                try:
                    value = float(trunc_string)
                except ValueError:
                    continue
                input_key_value_flags[flag] = value
                params.remove(param)
    

    last_num = min(len(params), last_num)
    if is_last: 
        small_params = []
        for i in range(-last_num, 0):
            small_params.append(params[i])
        params = small_params
        
    string = ' '.join(params)
    return(string, key_value_flags, input_key_value_flags)


# stopgap because twitchio v3 doesn't support spaces between prefix and command anymore...
def build_prefixes(chars: set) -> list:
    prefixes = []
    for ch in chars:
        prefixes.append(ch + " ")
        prefixes.append(ch)
    return prefixes