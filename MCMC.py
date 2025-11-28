import numpy as np
import random
import math
import time

alphabet = "abcdefghijklmnopqrstuvwxyz"

def decrypt(ciphertext, key):
    mapping = {key[i]: alphabet[i] for i in range(26)} # create a mapping dictionary from cipher letter to plain letter
    decrypted = [] # empty list to store the decrypted characters
    for a in ciphertext.lower():
        if a in alphabet:
            decrypted.append(mapping[a]) #append each decrypted character to the list
        else:
            decrypted.append(a)
    return "".join(decrypted) # join the list into a string and return it. 

def bigram_matrix(text):
    counts = np.zeros((26, 26)) # initialize a 26x26 matrix with all zeros

    filtered = [c for c in text.lower() if c in alphabet] # only take letters in the text we provide. The numbers and special characters were giving errors.

    for a, b in zip(filtered[:-1], filtered[1:]): # this zip function is similar to PuGofer's zip. It creates pairs of adjacent characters.
        i = ord(a) - ord('a')
        j = ord(b) - ord('a')
        counts[i, j] += 1 # for that particular value it updates by adding 1 every time it appears in the text.

    counts += 1 # to avoid zero rows. add one to each entry.
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = counts / row_sums

    return probs



def score_text(text, bigram_probs):
    filtered = [c for c in text.lower() if c in alphabet]

    log_prob = math.log(1/26) # initial probability for the first character, assuming uniform distribution

    for a, b in zip(filtered[:-1], filtered[1:]): # again using zip to create pairs of adjacent characters
        i = ord(a) - ord('a')
        j = ord(b) - ord('a')

        p = bigram_probs[i, j]

        if not (p > 0):  # needed this to avoid log(0) and getting -inf. ChatGPT helped debug this.
            p = 1e-12

        log_prob += math.log(p)

    return log_prob



def propose_new_key(key):
    key_list = list(key)
    i, j = random.sample(range(26), 2)
    key_list[i], key_list[j] = key_list[j], key_list[i] # "randomly" swap two letters
    return "".join(key_list)



def accept_probability(old_score, new_score):
    diff = new_score - old_score

    # Always accept improvements
    if diff >= 0:
        return True

    else:
        return random.random() < math.exp(diff) # this returns true or false because random.random() generates a number between 0 and 1.
                                                # Needed ChatGPTs help to figure this part out

  


def mcmc_step(ciphertext, key, old_score, bigram_probs):
    new_key = propose_new_key(key)
    decrypted = decrypt(ciphertext, new_key)
    new_score = score_text(decrypted, bigram_probs)

    if accept_probability(old_score, new_score):
        return new_key, new_score
    else:
        return key, old_score


 # uncomment this block for the MCMC decryption with random initial seeding
if __name__ == "__main__":

    with open("warandpeace.txt", "r", encoding="utf-8", errors="ignore") as f: # Looked up stackoverflow for this.
        corpus = f.read()

    bigram_probs = bigram_matrix(corpus)

    cipher = """bopuifs jnqpsubou rvftujpo up btl jt xiz epft uijt nfuipe pg bttjhojoh ofx lfzt fotvsf gbtu dpowfshfodf up uif usvf lfz gps dpnqbsjtpo csvuf gpsdjoh uijt xpvme sfrvjsf fwfo b tvqfs dpnqvufs bcmf up difdl usjmmjpot pg lfzt b tfdpoe bcpvu njmmjpo zfbst up difdl fwfsz lfz xifsfbt pvs bmhpsjuin uibu tnbsumz dipptft lfzt ublft b njovuf gsbdujpo pg uibu ujnf voefs b njovuf uif bmhpsjuin bdijfwft tjhojgjdboumz rvjdlfs tpmwf sbuft cfdbvtf ju fyqmplus qspqfsujft pg uif fohmjti mbohvbhf gps fybnqmf tjodf xf bsf difdljoh pvs tdpsft cbtfe po qbjst pg mfuufst bqqfbsjoh jo b tqfdjgjd psefs bt tppo bt b qbjs bt dpnnpo bt ui ps jo jt pvuqvuufe jo b efdszqufe ufyu uif bmhpsjuin cfhjot fyqmpsjoh uif lfzt xjui uiptf cjhsbnt gjyfe uifz bsf opu bduvbmmz gjyfe ju jt kvtu ijhimz vomjlfmz uibu uif bmhpsjuin tfmfdut b ofx lfz uibu dibohft uifn uijt sfevdft uibu tqbdf pg lfzt up cf difdlfe tjhojgjdboumz"""
    cipher = cipher.lower()

    seed_key = ''.join(random.sample(alphabet, 26))

    initial_decrypted = decrypt(cipher, seed_key)
    initial_score = score_text(initial_decrypted, bigram_probs)

    print("Initial key:", seed_key)
    print("Initial score:", initial_score)

    iterations = 0
    max_iterations = 200000

    best_key = seed_key
    best_score = initial_score
    start_time = time.perf_counter()
    while iterations < max_iterations:

        new_key, new_score = mcmc_step(cipher, seed_key, initial_score, bigram_probs)

        # accepted
        if new_key != seed_key:
            seed_key = new_key
            initial_score = new_score

        # track best
        if new_score > best_score:
            best_score = new_score
            best_key = new_key
            print("\n[New Best Score]", best_score)
            print("Best key:", best_key)
            print("Decrypted:", decrypt(cipher, best_key))

        iterations += 1

    print("\nFinal best decryption:")
    print(decrypt(cipher, best_key))
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"\n Total time taken: {total_time} seconds")



""" The algorithm is clearly finding the right decryption within 5-10 seconds but because my termination condition is actually the number of iterations  
total time taken is closer to a minute. Intialising with a manual seed will still not change this but might make a minute difference in how quickly it finds the right decryption."""




