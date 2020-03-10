import argparse
import sys

parser = argparse.ArgumentParser(description='Gaia parser.')
parser.add_argument('-o', '--output', help='generated code.', action='store', type=str)
parser.add_argument('-c', '--code', help='Gaia IR code', action='store', type=str)
args = parser.parse_args()

output_f = open(args.output, 'w')
code_f = open(args.code, 'r')

output_f.write('#include \"ops.h\"\n')
output_f.write('#include \"type.h\"\n')
output_f.write('#include \"scat_gather.h\"\n\n')

output_f.write('void ' + args.code.split('/')[-1][:-5] + '(scalar_t* INPUT, scalar_t* WEIGHT, scalar_t* OUTPUT,\n')
output_f.write('              int out_ch, int in_ch, int k_size, int in_h, int in_w,\n')
output_f.write('              int stride, int l_pad, int r_pad, int u_pad, int d_pad,\n')
output_f.write('              scalar_t* in_mem, scalar_t* wt_mem, scalar_t* ot_mem)\n')
output_f.write('{\n')

INPUT = ()
WEIGHT = ()
OUTPUT = ()

T_INPUT = []
T_WEIGHT = []
T_OUTPUT = []

def variable_parsing(var):
    p_var = var[:-2].split('(')
    p_var[0] = p_var[0].split('_')
    p_var[1] = p_var[1].split(',')
    return p_var

section = ''

while True:
    line = code_f.readline()
    if not line: break
    if line == '[var]\n': 
        print('variable section catch...')
        section = 'var'
        continue
    if line == '[text]\n': 
        print('text section catch...')
        section = 'text'
        continue
    if line == '\n': 
        continue
    if 'MEM' in line and ' ' not in line:
        print(line[:-1].split('(')[0] + ' catch...')
        continue
    if section == 'var':
        parsed_var = variable_parsing(line)
        if len(parsed_var[0]) == 2:
            if parsed_var[0][0] == 'INPUT':
                INPUT = tuple(map(int, parsed_var[1][1:]))
            elif parsed_var[0][0] == 'WEIGHT':
                WEIGHT = tuple(map(int, parsed_var[1][1:]))
            elif parsed_var[0][0] == 'OUTPUT':
                OUTPUT = tuple(map(int, parsed_var[1][1:]))
            else:
                print('Invalid untiled variable:  ' + parsed_var[0][0])
                sys.exit()
        else:
            if parsed_var[0][0] == 'INPUT':
                T_INPUT += [tuple(map(int, parsed_var[1][1:]))]
            elif parsed_var[0][0] == 'WEIGHT':
                T_WEIGHT += [tuple(map(int, parsed_var[1][1:]))]
            elif parsed_var[0][0] == 'OUTPUT':
                T_OUTPUT += [tuple(map(int, parsed_var[1][1:]))]
    elif section == 'text':
        ind = '  '
        parsed_code = line.split()
        if parsed_code[0] == 'LOAD':
            output_f.write('#if defined(CUDA) || defined(CUBLAS)\n')
            output_f.write(ind + 'tile_cpy4d_gpu(')
            if parsed_code[1][:2] == 'IN':
                output_f.write('in_mem, INPUT, ')
                index = int(parsed_code[2].split('_')[2])
                t_input = T_INPUT[index]
                output_f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, HOST_TO_DEVICE);\n'.format(t_input[0], INPUT[1], INPUT[2], INPUT[3], INPUT[4], t_input[1], t_input[2], t_input[3], t_input[4]))
            elif parsed_code[1][:2] == 'WT':
                output_f.write('wt_mem, WEIGHT, ')
                index = int(parsed_code[2].split('_')[2])
                t_weight = T_WEIGHT[index]
                output_f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, HOST_TO_DEVICE);\n'.format(t_weight[0], WEIGHT[1], WEIGHT[2], WEIGHT[3], WEIGHT[4], t_weight[1], t_weight[2], t_weight[3], t_weight[4]))
            elif parsed_code[1][:2] == 'OT':
                output_f.write('ot_mem, OUTPUT, ')
                index = int(parsed_code[2].split('_')[2])
                t_output = T_OUTPUT[index]
                output_f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, HOST_TO_DEVICE);\n'.format(t_output[0], OUTPUT[1], OUTPUT[2], OUTPUT[3], OUTPUT[4], t_output[1], t_output[2], t_output[3], t_output[4]))
            output_f.write('#else\n');
            output_f.write(ind + 'tile_cpy4d(')
            if parsed_code[1][:2] == 'IN':
                output_f.write('in_mem, INPUT, ')
                index = int(parsed_code[2].split('_')[2])
                t_input = T_INPUT[index]
                output_f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, HOST_TO_DEVICE);\n'.format(t_input[0], INPUT[1], INPUT[2], INPUT[3], INPUT[4], t_input[1], t_input[2], t_input[3], t_input[4]))
            elif parsed_code[1][:2] == 'WT':
                output_f.write('wt_mem, WEIGHT, ')
                index = int(parsed_code[2].split('_')[2])
                t_weight = T_WEIGHT[index]
                output_f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, HOST_TO_DEVICE);\n'.format(t_weight[0], WEIGHT[1], WEIGHT[2], WEIGHT[3], WEIGHT[4], t_weight[1], t_weight[2], t_weight[3], t_weight[4]))
            elif parsed_code[1][:2] == 'OT':
                output_f.write('ot_mem, OUTPUT, ')
                index = int(parsed_code[2].split('_')[2])
                t_output = T_OUTPUT[index]
                output_f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, HOST_TO_DEVICE);\n'.format(t_output[0], OUTPUT[1], OUTPUT[2], OUTPUT[3], OUTPUT[4], t_output[1], t_output[2], t_output[3], t_output[4]))
            output_f.write('#endif\n');
        elif parsed_code[0] == 'CONV':
            output_f.write(ind + '// ' + line)
            output_f.write('#if defined(CUDA) || defined(CUBLAS)\n')
            output_f.write(ind + 'conv2d_gpu(in_mem, wt_mem, ot_mem, ')
            t_input = T_INPUT[int(parsed_code[2].split('_')[2])]
            t_weight = T_WEIGHT[int(parsed_code[3].split('_')[2])]
            t_output = T_OUTPUT[int(parsed_code[1].split('_')[2])]
            stride = int(parsed_code[4])
            padding = tuple(parsed_code[5][1:-1].split(','))
            output_f.write(ind + '{}, {}, {}, {}, {}, {}, {}, {}, {}, {});\n'.format(t_output[2], t_input[2], t_weight[3], t_input[3], t_input[4], stride, padding[0], padding[1], padding[2], padding[3]))
            output_f.write('#else\n')
            output_f.write(ind + 'conv2d(in_mem, wt_mem, ot_mem, ')
            t_input = T_INPUT[int(parsed_code[2].split('_')[2])]
            t_weight = T_WEIGHT[int(parsed_code[3].split('_')[2])]
            t_output = T_OUTPUT[int(parsed_code[1].split('_')[2])]
            stride = int(parsed_code[4])
            padding = tuple(parsed_code[5][1:-1].split(','))
            output_f.write(ind + '{}, {}, {}, {}, {}, {}, {}, {}, {}, {});\n'.format(t_output[2], t_input[2], t_weight[3], t_input[3], t_input[4], stride, padding[0], padding[1], padding[2], padding[3]))
            output_f.write('#endif\n')
        elif parsed_code[0] == 'STORE':
            output_f.write('#if defined(CUDA) || defined(CUBLAS)\n')
            output_f.write(ind + 'tile_cpy4d_gpu(OUTPUT, ot_mem, ')
            index = int(parsed_code[1].split('_')[2])
            t_output = T_OUTPUT[index]
            output_f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, DEVICE_TO_HOST);\n'.format(t_output[0], OUTPUT[1], OUTPUT[2], OUTPUT[3], OUTPUT[4], t_output[1], t_output[2], t_output[3], t_output[4]))
            output_f.write('#else\n')
            output_f.write(ind + 'tile_cpy4d(OUTPUT, ot_mem, ')
            index = int(parsed_code[1].split('_')[2])
            t_output = T_OUTPUT[index]
            output_f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, DEVICE_TO_HOST);\n\n'.format(t_output[0], OUTPUT[1], OUTPUT[2], OUTPUT[3], OUTPUT[4], t_output[1], t_output[2], t_output[3], t_output[4]))
            output_f.write('#endif\n')

output_f.write('}')

output_f.close()
code_f.close()
