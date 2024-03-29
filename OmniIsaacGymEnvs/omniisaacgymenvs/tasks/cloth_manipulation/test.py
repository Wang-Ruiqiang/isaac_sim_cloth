import sys 
from queue import Queue

def step_one(lines):
    dish_type_num = int(lines[1])
    dish_num = []
    dish_rest_num = []
    dish_price = []
    for i in range(dish_type_num) :
        dish_message = str(lines[i + 2]).split(' ')
        dish_num.append(dish_message[0])
        dish_rest_num.append(dish_message[1])
        dish_price.append(dish_message[2])
    for i in range(dish_type_num + 2, len(lines)):
        dish_message = str(lines[i + dish_type_num + 2]).split(' ')


def step_two(lines):
    dish_type_num_micro_num = str(lines[1]).split(' ')
    dish_type_num = int(dish_type_num_micro_num[0])
    micro_num = int(dish_type_num_micro_num[1])
    using_micro_num = [0] * micro_num
    dish_num = []
    dish_rest_num = []
    dish_price = []
    wait_quene = Queue()
    for i in range(dish_type_num) :
        dish_message = str(lines[i + 2]).split(' ')
        dish_num.append(dish_message[0])
        dish_rest_num.append(dish_message[1])
        dish_price.append(dish_message[2])
    for i in range(dish_type_num + 2, len(lines)):
        dish_message = str(lines[i]).split(' ')
        if dish_message[0] == "received" :
            for i in range(len(using_micro_num)):
                if using_micro_num[i] == 0:
                    using_micro_num[i] = dish_message[3]
                    print(dish_message[3])
                    break
                if i == (len(using_micro_num) -1) :
                    print("wait")
                    wait_quene.put(dish_message[3])
        elif dish_message[0] == "complete":
            for i in range(len(using_micro_num)):
                if using_micro_num[i] == dish_message[1]:
                    if wait_quene.empty() :
                        print("ok")
                        using_micro_num[i] = 0
                    else :
                        now_num = wait_quene.get()
                        print("ok ", now_num)
                        using_micro_num[i] = now_num
                    break
                if i == (len(using_micro_num) -1) :
                    print("unexpected input")
    

def step_three(lines):
    print("in step three")


def step_four(lines) :
    print("in step four")

def main(lines):
    if lines[0] == "1":
        step_one(lines)
    elif lines[0] == "2":
        step_two(lines)
    elif lines[0] == "3":
        step_three(lines)
    else :
        step_four(lines)
    # for i, v in enumerate(lines) :
    #     print("line [{0}]: {1}".format(i, v))

if __name__ == '__main__':
    lines = []
    for l in sys.stdin:
        lines.append(l.strip('\r\n'))
    main(lines)
