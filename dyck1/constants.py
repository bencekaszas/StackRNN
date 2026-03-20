# Dyck-1 Task Constants
BATCH_SIZE = 64
SEQ_LENGTH = 60    # Maximum length of input prefix

TEST_SEQ_LENGTH = 200   
MAX_LEN = 2 * TEST_SEQ_LENGTH + 5

HIDDEN_DIM = 64
STACK_DEPTH = 300
LEARNING_RATE = 1e-4
STEPS = 12000
EVAL_FREQ = 500

# input (Dyck-1)
# 0: PAD, 1: '(', 2: ')', 3: '=', 4: EOS
DYCK_PAD, DYCK_OPEN, DYCK_CLOSE, DYCK_EQ, DYCK_EOS = 0, 1, 2, 3, 4
DYCK_VOCAB_SIZE = 5

# stack
# 0: NULL, 1: '(' (We only need to store the 'open' bracket)
DYCK_STACK_NULL, DYCK_STACK_OPEN = 0, 1
DYCK_STACK_VOCAB_SIZE = 2

# memory actions
ACT_NOOP, ACT_PUSH_0, ACT_PUSH_1, ACT_POP = 0, 1, 2, 3
# Note: For Dyck-1, we only use ACT_PUSH_0 (to push DYCK_STACK_OPEN) and ACT_POP
NUM_MEM_ACTIONS = 4

# controller states
DYCK_NUM_STATES = 8
