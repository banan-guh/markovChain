INVISIBLE = {0x034F, 0x200B, 0x200C, 0x200D, 0x200E, 0x200F, 0xFEFF, 0x00AD, }


def parse_flags(string, flags, is_last = False, last_num = 1):
    for char in string:
        if ord(char) in INVISIBLE:
            string = string.replace(char, '')
    
    params = string.split()
    
    key_value_flags = dict.fromkeys(flags, False)

    for flag in flags:
        if flag in params:
            key_value_flags[flag] = True
            params.remove(flag)

    last_num = min(len(params), last_num)
    if is_last: 
        small_params = []
        for i in range(-last_num, 0):
            small_params.append(params[i])
        params = small_params
        
    string = ' '.join(params)
    return(string, key_value_flags)