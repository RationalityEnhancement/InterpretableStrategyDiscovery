from RL2DT.formula_visualization import dnf2conj_list as d2l, unit_pred2expr
from hyperparams import CLEANED_PREDS as condied_preds
import re

## Change to obtain translation for road-trip planning task
MORTGAGE = True
## A collection of domain=specific words
ART = ' an ' #' a '
ART_C = ' a '
OBJ = 'interest rate' if MORTGAGE else 'hotel price'
MOD_OBJ = 'interest rate' if MORTGAGE else "hotels' prices"
CHOICE = 'mortgage' if MORTGAGE else 'route'
WHAT = ' interest' if MORTGAGE else ' price'
LVL = 'payment' if MORTGAGE else 'the trip'
SHORT = 'short-term' if MORTGAGE else 'proximate'
MID = 'mid-term' if MORTGAGE else 'mid'
LONG = 'long-term' if MORTGAGE else 'distant'
ACTED = 'clicked' if MORTGAGE else 'looked up'
ACT = 'Click' if MORTGAGE else 'Look up'
DONE = 'clicked' if MORTGAGE else 'looked up'
OBJ_PART = 'rate' if MORTGAGE else 'hotel'

def pred_dictt(pred, val, one_step=True):
    """
    High-level funciton to pass appropriate arguments for expression building
    functions.
    """
    not_ = False
    if 'not' == pred[:3] :
        if 'not(among' in pred: pred = pred[4:-1]
        not_ = True
    if pred[-1] == ')': pred= pred[:-1] + ' )'
    if pred[-2:] == ') ': pred = pred[:-2] + ' )'
    o = logic2words(pred, one_step=one_step, val=val)
    if not_: out = o.lower()
    else: out = o
    return(out)
    
def unit_pred2expr(predicate_str, unit=False, first_conj=False,
                   second_conj=False, special=False, one_step=True, until=False):
    """
    Convert a string being a predicate to an expression.
    
    For SIMPLE and GENERAL predicates from RL2DT.PLP.DSL
    """
    ## SIMPLE predicates
    if ' not' in predicate_str or 'not ' in predicate_str:
        real_pred = predicate_str[5:-1]
        not_ = 'not '
    elif ' not ' in predicate_str:
        real_pred = predicate_str[6:-1]
        not_ = 'not '
    elif 'not' in predicate_str:
        real_pred = predicate_str[4:-1]
        not_ = 'not '
    else:
        real_pred = predicate_str
        not_ = ''

    if re.match('\s{0,1}depth\(', real_pred):
        x = real_pred.split()[1]
        if one_step:
            rodz = ' a '
            s = ''
        else:
            rodz = ' the '
            s = 's'
        if '1' in x:
            x = 'short-term ' + OBJ
        elif '2' in x:
            x = 'mid-term ' + OBJ
        else:
            x = 'long-term ' + OBJ
        if not_ != '': 
            add = ' any ' + OBJ + s + ' but for'
        else:
            add = ''
        if first_conj or second_conj or unit:
            if one_step:
                on = ''#'represents'
            else:
                on = ''#'represent'
        ## 'that reside'+s+
        return add + rodz + x

    if 'on_highest' in real_pred:
        if one_step:
            s = ART + + OBJ + ' of'
            sp = ''
        else:
            s = ' the ' + OBJ + 's' + ' of'
            sp = 's'
        where = ' the most viable ' + CHOICE + sp
        if not_ != '':
            not_ = ' any but'
        if first_conj or unit:
            return s + not_ + where
        if second_conj:
            return s + not_ + where
        return s + not_ + where
    if '2highest' in real_pred:
        if one_step:
            s = ART + OBJ + ' of'
            sp = ''
        else:
            s = ' the ' + OBJ + 's' + ' of'
            sp = 's'
        where = ' the second most viable ' + CHOICE + sp
        if not_ != '':
            not_ = ' any but'
        if first_conj or unit:
            return s + not_ + where
        if second_conj:
            return s + not_ + where
        return s + not_ + where
        
    if 'are_branch' in real_pred:
        if not_ != '':
            message = 'whose long-term ' + OBJ + 's' + ' are not all ' + DONE
        else:
            message = 'whose long-term ' + OBJ + 's' + ' are all ' + DONE
        if first_conj or unit:
            if one_step:
                return ART + OBJ + ' of the ' + CHOICE + ' ' + message
            else:
                return ' the ' + OBJ + 's' + ' of ' + CHOICE + 's ' + message
        if second_conj:
            return message
        return message

    if 'is_leaf' in real_pred:
        if one_step:
            rodz = ' a '
            s = ''
        else:
            rodz = ' the '
            s = 's'
        what = rodz + ' long-term ' + OBJ + s
        if not_ != '':
            not_ = 'any but for '
        if first_conj or unit:
            if one_step:
                return not_ + what
            else:
                return not_ + what
        if second_conj:
            if one_step:
                return not_ + what
            else:
                return not_ + what     
        return not_ + what
    if 'is_root' in real_pred:
        if one_step:
            rodz = ' a '
            s = ''
        else:
            rodz = ' the '
            s = 's'
        what = rodz + ' short-term ' + OBJ + s
        if not_ != '':
            not_ = 'any but for '
        if first_conj or unit:
            if one_step:
                return not_ + what
            else:
                return not_ + what
        if second_conj:
            if one_step:
                return not_ + what
            else:
                return not_ + what     
        return not_ + what
    if 'is_max_in' in real_pred:
        if not_ != '':
            not_ = 'with none '
        else:
            not_ = 'with one '
        if one_step:
            start = ART + OBJ + ' of the ' + CHOICE + ' ' + not_ + ' of the ' + OBJ + 's being the most viable'
        else:
            start = ' the ' + OBJ + 's of ' + CHOICE + ' ' + not_ + ' of the ' + OBJ + 's being the most viable' 
        return start
    if 'is_2max_in' in real_pred:
        if not_ != '':
            not_ = 'with less than 2 '
        else:
            not_ = 'with 2 '
        if one_step:
            start = ART + OBJ + ' of the ' + CHOICE + ' ' + not_ + ' of the ' + OBJ + 's being the most viable'
        else:
            start = ' the ' + OBJ + 's of ' + CHOICE + ' ' + not_ + ' of the ' + OBJ + 's being the most viable' 
        return start
    if 'is_child' in real_pred:
        if one_step:
            start = ' the ' + OBJ + 's which '
        else:
            start = ART + OBJ + ' which '
        if not_ == '':
            which = 'positive'
        else:
            which = 'negative or unobserved'
        if first_conj or unit:
            if one_step:
                return start+'comes directly sooner than' + ART + OBJ + ' with a '+which+' value'
            else:
                return start+'come directly sooner than ' + OBJ + 's with '+which+' values'
        if second_conj:
            if one_step:
                return start+'comes directly sooner than' + ART + OBJ + ' with a '+which+' value'
            else:
                return start+'come directly sooner than ' + OBJ + ' with '+which+' values'
        return start+'come directly sooner than ' + OBJ + ' with '+which+' values'
    if 'is_parent' in real_pred:
        if one_step:
            start = ' the ' + OBJ + 's which '
        else:
            start = ART + OBJ + ' which '
        if not_ == '':
            which = 'positive'
        else:
            which = 'negative or unobserved'
            if one_step:
                return start+'comes directly later than' + ART + OBJ + ' with a '+which+' value'
            else:
                return start+'come directly later than ' + OBJ + 's with '+which+' values'
        if second_conj:
            if one_step:
                return start+'comes directly later than' + ART + OBJ + ' with a '+which+' value'
            else:
                return start+'come directly later than ' + OBJ + ' with '+which+' values'
        return start+'come directly later than ' + OBJ + ' with '+which+' values'
    if 'highest' in real_pred or 'smallest' in real_pred:
        finish = 'the most viable' if 'highest' in real_pred else 'the least viable'
        if 'leaf' in real_pred:
            #which = '48' if 'highest' in real_pred else '-48'
            if not_ != '':
                not_ = 'any but '
            if first_conj or unit:
                if one_step:
                    return ART + OBJ + ' of the ' + CHOICE + ' whose possible long-term ' + OBJ + ' is ' + not_ + finish
                else:
                   return 'the ' + OBJ + 's of ' + CHOICE + 's whose possible long-term ' + OBJ + 's are ' + not_ + finish
            if second_conj:
                if one_step:
                    return ART + OBJ + ' of the ' + CHOICE + ' whose possible long-term ' + OBJ + ' is ' + not_ + finish
                else:
                    return 'the ' + OBJ + 's of ' + CHOICE + 's whose possible long-term ' + OBJ + 's are ' + not_ + finish
            return ART + OBJ + ' of the ' + CHOICE + ' whose possible long-term ' + OBJ + ' is ' + not_ + finish
        if 'root' in real_pred:
            #which = '4' if 'highest' in real_pred else '-4'
            if not_ != '':
                not_ = 'any but '
            if first_conj or unit:
                if one_step:
                    return ART + OBJ + ' of the ' + CHOICE + ' whose possible short-term ' + OBJ + ' is ' + not_ + finish
                else:
                   return 'the ' + OBJ + 's of ' + CHOICE + 's whose possible short-term ' + OBJ + 's are ' + not_ + finish
            if second_conj:
                if one_step:
                    return ART + OBJ + ' of the ' + CHOICE + ' whose possible short-term ' + OBJ + ' is ' + not_ + finish
                else:
                    return 'the ' + OBJ + 's of ' + CHOICE + 's whose possible short-term ' + OBJ + 's are ' + not_ + finish
            return ART + OBJ + ' of the ' + CHOICE + ' whose possible short-term ' + OBJ + ' is ' + not_ + finish
        if 'child' in real_pred:
            if not_ != '':
                not_ = 'any but '
            if first_conj or unit:
                if one_step:
                    return ART + OBJ + ' which comes directly later than the ' + OBJ + ' with ' + not_ + finish + WHAT + ' for its stage of ' + LVL
                else:
                   return 'the ' + OBJ + 's which comes directly later than ' + OBJ + 's with ' + not_ + finish + ' for their stage of ' + LVL
            if second_conj:
                if one_step:
                    return ART + OBJ + ' which comes directly laternthan the ' + OBJ + ' with ' + not_ + finish + ' for its stage of ' + LVL
                else:
                    return 'the ' + OBJ + 's which comes directly later than ' + OBJ + 's with ' + not_ + finish + ' for their stage of ' + LVL
            return ART + OBJ + ' which comes directly later than the ' + OBJ + ' with ' + not_ + finish + ' for its stage of ' + LVL
        if 'parent' in real_pred:
            if not_ != '':
                not_ = 'any but '
            if first_conj or unit:
                if one_step:
                    return ART + OBJ + ' which comes directly sooner than the ' + OBJ + ' with ' + not_ + finish + WHAT + ' for its stage of ' + LVL
                else:
                   return 'the ' + OBJ + 's which comes directly sooner than ' + OBJ + 's with ' + not_ + finish + ' for their stage of ' + LVL
            if second_conj:
                if one_step:
                    return ART + OBJ + ' which comes directly sooner than the ' + OBJ + ' with ' + not_ + finish + ' for its stage of ' + LVL
                else:
                    return 'the ' + OBJ + 's which comes directly sooner than ' + OBJ + 's with ' + not_ + finish + ' for their stage of ' + LVL
            return ART + OBJ + ' which comes directly sooner than the ' + OBJ + ' with ' + not_ + finish + ' for its stage of ' + LVL

    if 'is_observed' in real_pred:
        if not_ != '':
            not_ = "have not "
        else:
            not_ = 'have '
        return ' that you ' + not_ + ACTED + ' yet'

    ## GENERAL predicates
    if 'is_none' in real_pred:
        return 'no ' + OBJ + ' is ' + DONE + ' yet'
    if 'is_all' in real_pred:
        return 'all the ' + OBJ + 's are ' + DONE
    if 'is_path' in real_pred:
        if not_ != '':
            return 'no ' + CHOICE + ' has all ' + OBJ + 's ' + DONE
        return 'there exists' + ART_C + CHOICE + ' whose all ' + OBJ + 's are ' + DONE
    if 'is_parent_depth' in real_pred:
        if not_ != '':
            return 'some of the later ' + OBJ + 's are un' + DONE
        return 'all the later ' + OBJ + 's are ' + DONE
    if 'is_previous_observed' in real_pred:
        if not(until):
            e = 'the most recently uncovered information regards'
        else:
            e = 'you have encountered'
        if 'positive' in real_pred:
            if not_ != '':
                return e + ART + OBJ + ' with a negative value'
            return e + ART + OBJ + ' with a positive value'
        if 'parent' in real_pred:
            if not_ != '':
                return 'you have not ' + ACTED + ' a sooner ' + OBJ + ' directly preceding this ' + OBJ
            return 'you have ' + ACTED + ' a sooner ' + OBJ + ' directly preceding this' + OBJ
        if 'sibling' in real_pred:
            if not_ != '':
                return 'you have not ' + ACTED + ART + OBJ + ' on the same stage of ' + LVL
            return 'you have ' + ACTED + ART + OBJ + ' on a different stage of ' + LVL
        if 'max_level' in real_pred:
            w = real_pred.split()
            num = w[1]
            if '3' in num:
                if not_ != '':
                    return e + ' something else than the lowest possible ' + LONG + ' ' + MOD_OBJ
                return e + ' the lowest possible ' + LONG + ' ' + MOD_OBJ
            elif '2' in num:
                if not_ != '':
                    return e + ' something else than the lowest possible ' + MID + ' ' + MOD_OBJ
                return e + ' the lowest possible ' + MID + ' ' + MOD_OBJ
            elif '1' in num:
                if not_ != '':
                    return e + ' something else than the lowest possible ' + SHORT + ' ' + MOD_OBJ
                return e + ' the lowest possible ' + SHORT + ' ' + MOD_OBJ
        if 'min_level' in real_pred:
            w = real_pred.split()
            num = w[1]
            if '3' in num:
                if not_ != '':
                    return e + ' something else than the highest possible ' + LONG + ' ' + MOD_OBJ
                return e + ' the highest possible ' + LONG + ' ' + MOD_OBJ
            elif '2' in num:
                if not_ != '':
                    return e + ' something else than the highest possible ' + MID + ' ' + MOD_OBJ
                return e + ' the highest possible ' + MID + ' ' + MOD_OBJ
            elif '1' in num:
                if not_ != '':
                    return e + ' something else than the highest possible ' + SHORT + ' ' + MOD_OBJ
                return e + ' the highest possible ' + SHORT + ' ' + MOD_OBJ
        if 'max_nonleaf' in real_pred:
            if not_ != '':
                return e + ' something else than the lowest possible ' + MID + ' ' + MOD_OBJ
            return e + ' the lowest possible ' + MID + ' ' + MOD_OBJ
        if 'max_root' in real_pred:
            if not_ != '':
                return e + ' something else than the highest possible ' + SHORT + ' ' + MOD_OBJ
            return e + ' the highest possible short-term ' + SHORT + ' ' + MOD_OBJ
        if 'max' in real_pred:
            if not_ != '':
                return e + ' something else than the lowest possible ' + OBJ
            return e + ' the lowest possible ' + OBJ
        if 'min' in real_pred:
            if not_ != '':
                return e + ' something else than the highest possible ' + OBJ
            return e + ' the highest possible ' + OBJ
    if 'is_positive' in real_pred:
        if not_ != '':
            return 'neither of the ' + DONE + ' ' + OBJ + ' has a positive value'
        return 'there is' + ART + OBJ +' with a positive value'
    if 'are_leaves' in real_pred:
        if not_ != '':
            return 'some of the ' + LONG + ' ' +  MOD_OBJ + 's are un' + DONE
        return 'all the ' + LONG + ' ' +  MOD_OBJ + 's are ' + DONE
    if 'are_roots' in real_pred:
        if not_ != '':
            return 'some of the ' + SHORT + ' ' + MOD_OBJ + 's are un' + DONE
        return 'all the ' + SHORT + ' ' +  MOD_OBJ + 's are ' + DONE
    if 'previous_observed_depth' in real_pred:
        w = real_pred.split()
        num = w[1]
        if '1' in num:
            num = SHORT + ' ' +  MOD_OBJ + 's'
        elif '2' in num:
            num = MID + ' ' + MOD_OBJ + 's'
        else:
            num = LONG + ' ' + MOD_OBJ + 's'
        if not_ != '':
            return 'the most recently considered information regarded any but for ' + num
        return 'the most recently considered information regarded ' + num
    if 'count' in real_pred:
        w = real_pred.split()
        num = w[1]
        if num[-1] == ')': num = num[:-1]
        if num == 1:
            if not_ != '':
                return 'no ' + OBJ + ' is ' + DONE + ' yet'
            else:
                return 'there is at least 1 ' + DONE + ' ' + OBJ
        if not_ != '':
            return 'there are less than ' + num + ' ' + DONE + ' ' + OBJ + 's'
        return 'there are at least ' + num + ' ' + DONE + ' ' + OBJ + 's'
    if 'termination' in real_pred:
        w = real_pred.split()
        num = w[1]
        if num[-1] == ')': num = num[:-1]
        if not_ != '':
            return 'the lowest available ' + CHOICE + ' renders ' + OBJ + 's with a summed value different than ' + num
        return 'the lowest available ' + CHOICE + ' renders ' + OBJ + 's with a summed value of  ' + num

    return predicate_str
    
def pred2expr(predicate_str, unit=False, conjunction=False, one_step=False, until=False):
    """
    Join expressions of two SIMPLE predicates or output an expression for one, if
    there is only one.
    """
    if ' or ' in predicate_str:
        p1, p2 = predicate_str.split(' or ')
        p1 = p1[1:]
        p2 = p2[:-1]
        expr = unit_pred2expr(p1, first_conj=True, until=until) \
                   + ' or ' + unit_pred2expr(p2,second_conj=True, until=until)
    elif ' and ' in predicate_str:
        p1, p2 = predicate_str.split(' and ')
        if p1 == p2:
            expr = unit_pred2expr(p1, unit=True, one_step=one_step)
        else:
            if 'is_observed' in p1: 
                if 'is_root' in p2 or 'is_leaf' in p2 or 'depth' in p2:
                    expr = unit_pred2expr(p2, first_conj=True, one_step=one_step) \
                           + ' ' + unit_pred2expr(p1, special=True, one_step=one_step)
                else:
                    node = ' out of those ' + OBJ + 's that '
                    expr = unit_pred2expr(p2, first_conj=True, one_step=one_step) \
                           + node + unit_pred2expr(p1, special=True, one_step=one_step)
            elif 'is_observed' in p2: 
                if 'is_root' in p1 or 'is_leaf' in p1 or 'depth' in p1:
                    expr = unit_pred2expr(p1, first_conj=True, one_step=one_step) \
                           + ' ' + unit_pred2expr(p2, special=True, one_step=one_step)
                else:
                    node = ' out of those ' + OBJ + 's that '
                    expr = unit_pred2expr(p1, first_conj=True, one_step=one_step) \
                           + node + unit_pred2expr(p2, special=True, one_step=one_step)
            else:
                expr = unit_pred2expr(p1, first_conj=True, one_step=one_step) + \
                       ' and ' + unit_pred2expr(p2,second_conj=True, one_step=one_step)
    else:
        expr = unit_pred2expr(predicate_str, one_step=one_step, until=until, unit=True)
    return expr
    
def among_pred2expr_neg(predicate_str, prev, all_=False, one_step=True):
    """
    Convert a string being a predicate to an expression.
    
    For AMONG/ALL predicates from RL2DT.PLP.DSL
    """
    add = ' described by the point above'
    add = pred2expr(prev, one_step=False)
    if 'has_smallest' in predicate_str:
        if all_:
            return 'are the ' + OBJ + 's on the same stage of ' + LVL
        if 'the' == add[:3]:
            if one_step:
                if MORTGAGE:
                    return '\n - the most ' + SHORT + ' ' + MOD_OBJ + ' out of ' + add
                return '\n - the' + WHAT + ' of the most ' + SHORT + ' ' + OBJ_PART + 's'
            if MORTGAGE:
                return '\n - the most ' + SHORT + ' ' + MOD_OBJ + 's out of ' + add
            return '\n -' + WHAT + 's of the most ' + SHORT + ' ' + OBJ_PART + 's out of ' + add
        else:
            if one_step:
                if MORTGAGE:
                    return '\n - the most ' + SHORT + ' ' + MOD_OBJ + ' ' + add
                return '\n - the' + WHAT + ' of the most ' + SHORT + ' ' + OBJ_PART + ' ' + add
            if MORTGAGE:
                return '\n - the most ' + SHORT + ' ' + MOD_OBJ + 's ' + add
            return '\n - the' + WHAT + 's of the most ' + SHORT + ' ' + OBJ_PART + 's ' + add
    if 'has_largest' in predicate_str:
        if all_:
            return 'are the ' + OBJ + 's on the same stage of ' + LVL
        if 'the' == add[:3]:
            if one_step:
                if MORTGAGE:
                    return '\n - the most ' + LONG + ' ' + MOD_OBJ + ' out of ' + add
                return '\n - the' + WHAT + ' of the most ' + LONG + ' ' + OBJ_PART + 's'
            if MORTGAGE:
                return '\n - the most ' + LONG + ' ' + MOD_OBJ + 's out of ' + add
            return '\n -' + WHAT + 's of the most ' + LONG + ' ' + OBJ_PART + 's out of ' + add
        else:
            if one_step:
                if MORTGAGE:
                    return '\n - the most ' + LONG + ' ' + MOD_OBJ + ' ' + add
                return '\n - the' + WHAT + ' of the most ' + LONG + ' ' + OBJ_PART + ' ' + add
            if MORTGAGE:
                return '\n - the most ' + LONG + ' ' + MOD_OBJ + 's ' + add
            return '\n - the' + WHAT + 's of the most ' + LONG + ' ' + OBJ_PART + 's ' + add
        
    if 'the' != add[:3]:
        add = 'the ' + OBJ + 's that ' + add
    if 'has_lowest_path' in predicate_str:
        if all_:
            return 'are ' + OBJ +'s ' 'of ' + CHOICE + 's with the same value'     
        if one_step:
            return "\n - " + ART + OBJ + " of the " + CHOICE + " with the lowest possible " + OBJ + "s among " + add
        return "\n - the " + OBJ + "s of " + CHOICE + "s with the lowest possible " + OBJ + "s among " + add
    
    expr = predicate_str
    if 'has_parent_highest' in predicate_str:
        if one_step:
            expr = "\n - among " + add + ", this " + OBJ + " " + \
                   "comes directly later than the lowest possible " + OBJ
                   
        else:
            expr = "\n - among " + add + ", these outcomes " + \
                   "come directly later than the lowest possible " + OBJ + "s"
        if all_:
            return 'come directly later than ' + OBJ + 's with the same ' + OBJ + 's'
    if 'has_parent_smallest' in predicate_str:
        if one_step:
            expr = "\n - among " + add + ", this " + OBJ + " " + \
                   "comes directly later than the highest possible " + OBJ
                   
        else:
            expr = "\n - among " + add + ", these outcomes " + \
                   "come directly later than the highest possible " + OBJ + "s"
        if all_:
            return 'come directly later than ' + OBJ + 's with the same ' + OBJ + 's'
    if 'has_child_highest' in predicate_str:
        if one_step:
            expr = "\n - among " + add + ", this " + OBJ + " " + \
                   "comes directly sooner than the lowest possible " + OBJ
                   
        else:
            expr = "\n - among " + add + ", these outcomes " + \
                   "come directly sooner than the lowest possible " + OBJ + "s"
        if all_:
            return 'come directly sooner than ' + OBJ + 's with the same ' + OBJ + 's'
    if 'has_parent_smallest' in predicate_str:
        if one_step:
            expr = "\n - among " + add + ", this " + OBJ + " " + \
                   "comes directly sooner than the highest possible " + OBJ
                   
        else:
            expr = "\n - among " + add + ", these outcomes " + \
                   "come directly sooner than the highest possible " + OBJ + "s"
        if all_:
            return 'come directly sooner than ' + OBJ + 's with the same ' + OBJ + 's'
    if 'has_leaf_highest' in predicate_str:
        if one_step:
            expr = "\n - among " + add + ", this " + OBJ + \
                   "belongs to the " + CHOICE + " with the lowest possible " + LONG + ' ' + MOD_OBJ
                   
        else:
            expr = "\n - among " + add + ", these " + OBJ + "s" + \
                   "belong to the " + CHOICE + "s with the lowest possible " + LONG + ' ' + MOD_OBJ + "s"
        if all_:
            return 'belong to ' + CHOICE + 's whose ' + LONG + ' ' + MOD_OBJ + 's have the same value'
    if 'has_leaf_smallest' in predicate_str:
        if one_step:
            expr = "\n - among " + add + ", this " + OBJ + \
                   "belongs to the " + CHOICE + " with the highest possible " + LONG + ' ' + MOD_OBJ
                   
        else:
            expr = "\n - among " + add + ", these " + OBJ + "s" + \
                   "belong to the " + CHOICE + "s with the highest possible " + LONG + ' ' + MOD_OBJ + "s"
        if all_:
            return 'belong to ' + CHOICE + 's whose ' + LONG + ' ' + MOD_OBJ + 's have the same value'
    if 'has_root_highest' in predicate_str:
        if one_step:
            expr = "\n - among " + add + ", this " + OBJ + \
                   "belongs to the " + CHOICE + " with the lowest possible " + SHORT + ' ' + MOD_OBJ
                   
        else:
            expr = "\n - among " + add + ", these " + OBJ + "s" + \
                   "belong to the " + CHOICE + "s with the lowest possible " + SHORT + ' ' + MOD_OBJ + "s"
        if all_:
            return 'belong to ' + CHOICE + 's whose ' + LONG + ' ' + MOD_OBJ + 's have the same value'
    if 'has_root_smallest' in predicate_str:
        if one_step:
            expr = "\n - among " + add + ", this " + OBJ + \
                   "belongs to the " + CHOICE + " with the highest possible " + SHORT + ' ' + MOD_OBJ
                   
        else:
            expr = "\n - among " + add + ", these " + OBJ + "s" + \
                   "belong to the " + CHOICE + "s with the highest possible " + SHORT + ' ' + MOD_OBJ + "s"
        if all_:
            return 'belong to ' + CHOICE + 's whose ' + SHORT + ' ' + MOD_OBJ + 's have the same value'
    return expr
    
def among_pred2expr_pos(predicate_str, prev, all_=False, one_step=True):
    """
    Convert a string being a predicate to an expression.
    
    For AMONG/ALL predicates from RL2DT.PLP.DSL
    """
    add = ' ' + OBJ + 's described by the point above'
    add = pred2expr(prev, one_step=False)
    if 'has_smallest' in predicate_str:
        if all_:
            return 'are the same-term outcomes'
        if 'the' == add[:3]:
            if one_step:
                if MORTGAGE:
                    return 'the most ' + SHORT + ' ' + MOD_OBJ + ' out of ' + add
                return 'the' + WHAT + ' of the most ' + SHORT + ' ' + OBJ_PART + ' out of ' + add
            if MORTGAGE:    
                return 'the most ' + SHORT + ' ' + MOD_OBJ + 's out of ' + add
            return 'the' + WHAT + 's of the most ' + SHORT + ' ' + OBJ_PART + 's out of ' + add
        else:
            if one_step:
                if MORTGAGE:
                    return 'the most ' + SHORT + ' ' + MOD_OBJ + add
                return 'the' + WHAT + ' of the most ' + SHORT + ' ' + OBJ_PART + add
            if MORTGAGE:
                return 'the most ' + SHORT + ' ' + MOD_OBJ + 's' + add
            return 'the' + WHAT + 's of the most ' + SHORT + ' ' + OBJ_PART + 's' + add
    if 'has_largest' in predicate_str:
        if all_:
            return 'are the same-term outcomes'
        if 'the' == add[:3]:
            if one_step:
                if MORTGAGE:
                    return 'the most ' + LONG + ' ' + MOD_OBJ + ' out of ' + add
                return 'the' + WHAT + ' of the most ' + LONG + ' ' + OBJ_PART + ' out of ' + add
            if MORTGAGE:    
                return 'the most ' + LONG + ' ' + MOD_OBJ + 's out of ' + add
            return 'the' + WHAT + 's of the most ' + LONG + ' ' + OBJ_PART + 's out of ' + add
        else:
            if one_step:
                if MORTGAGE:
                    return 'the most ' + LONG + ' ' + MOD_OBJ + add
                return 'the' + WHAT + ' of the most ' + LONG + ' ' + OBJ_PART + add
            if MORTGAGE:
                return 'the most ' + LONG + ' ' + MOD_OBJ + 's' + add
            return 'the' + WHAT + 's of the most ' + LONG + ' ' + OBJ_PART + 's' + add

    if 'the' != add[:3]:
        add = 'the ' + OBJ + 's that ' + add
    if 'has_lowest_path' in predicate_str:
        if all_:
            return 'are ' + OBJ +'s ' 'of ' + CHOICE + 's with the same value'     
        if one_step:
            return pred2expr(prev, one_step=True) + " that belongs to the " + CHOICE + " with the lowest " + OBJ + "s"
        return add + " that belongs to the " + CHOICE + " with the lowest " + OBJ + "s"
    
    expr = predicate_str
    if 'has_parent_highest' in predicate_str:
        if one_step:
            expr = "one of " + add + " that come directly later than the lowest " + OBJ
                   
        else:
            expr = add + " that come directly later than the lowest possible " + OBJ + "s there"
        if all_:
            return 'come directly later than ' + OBJ + 's with the same value'
    if 'has_parent_smallest' in predicate_str:
        if one_step:
            expr = "one of " + add + " that come directly later than the highest " + OBJ
                   
        else:
            expr = add + " that come directly later than the highest possible " + OBJ + "s there"
        if all_:
            return 'come directly later than ' + OBJ + 's with the same value'
    if 'has_child_highest' in predicate_str:
        if one_step:
            expr = "one of " + add + " that come directly sooner than the lowest " + OBJ
                   
        else:
            expr = add + " that come directly sooner than the lowest possible " + OBJ + "s there"
        if all_:
            return 'come directly sooner than ' + OBJ + 's with the same value'
    if 'has_child_smallest' in predicate_str:
        if one_step:
            expr = "one of " + add + " that come directly sooner than the highest " + OBJ
                   
        else:
            expr = add + " that come directly sooner than the highest possible " + OBJ + "s there"
        if all_:
            return 'come directly sooner than ' + OBJ + 's with the same value'
    if 'has_leaf_highest' in predicate_str:
        if one_step:
            expr = "one of " + add + " that belong to the " + CHOICE + " with the lowest " + LONG + OBJ
                   
        else:
            expr = add + " that belong to " + CHOICE + "s with the lowest " + LONG + OBJ + "s"
        if all_:
            return 'belong to ' + CHOICE + ' whose ' + LONG + OBJ + 's have the same value'
    if 'has_leaf_smallest' in predicate_str:
        if one_step:
            expr = "one of " + add + " that belong to the " + CHOICE + " with the highest " + LONG + OBJ
                   
        else:
            expr = add + " that belong to " + CHOICE + "s with the highest " + LONG + OBJ + "s"
        if all_:
            return 'belong to ' + CHOICE + ' whose ' + LONG + OBJ + 's have the same value'
    if 'has_root_highest' in predicate_str:
        if one_step:
            expr = "one of " + add + " that belong to the " + CHOICE + " with the lowest " + SHORT + OBJ
                   
        else:
            expr = add + " that belong to " + CHOICE + "s with the lowest " + SHORT + OBJ + "s"
        if all_:
            return 'belong to ' + CHOICE + ' whose ' + LONG + OBJ + 's have the same value'
    if 'has_root_smallest' in predicate_str:
        if one_step:
            expr = "one of " + add + " that belong to the " + CHOICE + " with the highest " + SHORT + OBJ
                   
        else:
            expr = add + " that belong to " + CHOICE + "s with the highest " + SHORT + OBJ + "s"
        if all_:
            return 'belong to ' + CHOICE + ' whose ' + LONG + OBJ + 's have the same value'
    return expr
    
def logic2words(pred, val, one_step=True):
    """
    Convert a string being a predicate to an expression.
    
    For all the predicates from PLP.DSL
    """
    if re.match('among', pred):
        beg = 'it ' if one_step else "they "
        if re.search(':', pred):
            among_pred = re.search(':.+ \)', pred).group(0)[2:-2]
            list_preds = re.search('.+(?= :)', pred[6:]).group(0)
            what = pred2expr(list_preds, one_step=one_step)
            print(pred)
            print(val)
            if val == 'neg':
                expr = '\n - ' + beg + what + among_pred2expr_neg(among_pred, 
                                                                  list_preds, 
                                                                  one_step=one_step)
            elif val == 'pos':
                expr = among_pred2expr_pos(among_pred, 
                                           list_preds, 
                                           one_step=one_step)
                
        else:
            list_preds = pred[6:-2]
            p2e = pred2expr(list_preds, 
                            conjunction=True, 
                            one_step=one_step)
            p2e = p2e.split(' and ')
            if len(p2e) == 2:
                expr = '\n - ' + p2e[0] + '\n - ' + p2e[1]
            else:
                expr = p2e[0]
                   
            
    elif re.match('all_', pred) or re.match('not\(all_', pred):
        among_pred = re.search(':.+ \)', pred).group(0)[2:-2]
        list_preds = re.search('.+(?= :)', pred[5:]).group(0)
        if 'not(' == pred[:4]:
            beg = 'only part of '
        else:
            beg = 'all '
        expr = beg + pred2expr(list_preds, one_step=False) + ' ' + \
               among_pred2expr_pos(among_pred, list_preds, all_=True)

    elif not(any([pred in c for c in condied_preds])):
        beg = 'it ' if one_step else "they "
        expr = '\n - ' + beg + unit_pred2expr(pred, unit=True, one_step=one_step) ## więcej niże jedno niech się pojawi it/theyzs
    else:
        expr = unit_pred2expr(pred, unit=True, one_step=one_step)

    return expr
                  
def order_dnf(dnf_list):
    """
    Order a list of predicates starting from positive ones, then predicates that
    contain the preicate 'all_', and finally negated predicates.
    """
    dos, donts, alls = [], [], []
    for pred in dnf_list:
        if any([pred in cp for cp in condied_preds]) or 'all_' is pred:
            alls.append(pred)
        elif pred[:3] == 'not':
            donts.append(pred)
        else:
            dos.append(pred)
    return dos + alls + donts
    
def dnf2command(dnf_string, one_step=False):
    """
    Turn a DNF formula into a command in natural language using functions that 
    translate predicates into small expressions depending on the structure of
    the DNF.
    """
    preds = d2l(dnf_string,paran=False)[0]
    preds = order_dnf(preds)
    full = ''
    num_all, num_click, num_dont = 0,0,0

    for num, sub_pred in enumerate(preds):
                
        if any([sub_pred in cp for cp in condied_preds]) or 'all_' in sub_pred:
            if num_click == 0 and num_all == 0 and num_dont <= 1:
                full += ACT + " random " + OBJ + ". "
            num_all += 1
            if num_all > 1: beg = ''
            else:
                if one_step:
                    beg = '\n\nDo that under the condition that:'
                else: 
                    beg = '\n\nRepeat this until:'
            full += beg + '\n - '+ pred_dictt(sub_pred, val='neg', one_step=one_step)
            if not(any([sub_pred in cp for cp in condied_preds])): full += ' '
            
        elif sub_pred[:3] == 'not':
            num_dont += 1
            next = pred_dictt(sub_pred, val='neg', one_step=one_step)
            if num_click == 0 and num_all == 0 and num_dont == 1:
                full += ACT + " random " + OBJ + ". "
            if full[-1] != ' ': full += ' '
            if num_dont > 1:
                full += next
            else:
                full += '\n\nDo not ' + ACT.lower() + ':' + next

        else:
            num_click += 1
            if 'True' in sub_pred:
                if one_step:
                    full += ACT + " random " + OBJ + " or stop planning. "
                else:
                    full += 'Stop planning right away or ' + ACT.lower() + ' some random ' + OBJ + 's and then stop planning. '
            else:
                next = pred_dictt(sub_pred, val='pos', one_step=one_step)
                if num_click > 1:
                    full += '\n - ' + next
                else:
                    full += ACT + ' ' + next
    if full[-1] != ' ': full += ' ' 
    return full
    
def procedure2text(procedure_str):
    """
    Transform a procedural description/formula into a natural language text.
    
    Treat each DNF before AND NEXT as a step with number i, with i=1 for the
    first DNF, etc.
    
    Add appropriate suffixes to the comand if the DNF is followed by UNTIL or
    UNLESS.
    """
    if 'False' in procedure_str:
        txt = 'Do not ' + ACT.lower() + ' anything.'
        print("\n\n        Whole instructions: \n\n{}".format(txt))
        return txt
    if 'None' in procedure_str:
        txt = 'NO INSTRUCTION FOUND. (Treated as the no-planning strategy).'
        print("\n\n        Whole instructions: \n\n{}".format(txt))
        return txt
    txt = ''
    split = procedure_str.split('\n\nLOOP FROM ')
    if len(split) == 2:
        body, go_to = split
    else:
        body, go_to = split[0], None
    dnfs = body.split(' AND NEXT ')
    id_dict = {}
    for i, dnf in enumerate(dnfs):
        split_dnf = dnf.split(' UNTIL ')
        split_dnf2 = dnf.split(' UNLESS ')
        txt += str(i+1) + '. '
        if len(split_dnf) == 2 and len(split_dnf2) == 1:
            dnf_cond, until = split_dnf
            txt += dnf2command(dnf_cond)
            until_cond = 'until ' + pred2expr(until, until=True) if 'IT' not in until \
                         else "as long as possible"
            if txt[-1] == ' ':
                txt = txt[:-1] + '.'
            txt += "\n\nRepeat this step " + until_cond + ".\n\n(s)"
            #txt += until_cond + ".\n\n(s)"
        elif len(split_dnf) == 1 and len(split_dnf2) == 1:
            dnf_cond = split_dnf[0]
            txt += dnf2command(dnf_cond, one_step=True) + "\n\n(s)"
        else:# len(split_dnf) == 1 and len(split_dnf2) == 2:
            dnf_cond, unless = split_dnf2
            txt += 'Unless ' + pred2expr(unless, until=True) + \
                   ', in which case stop at the previous step, '
            txt += dnf2command(dnf_cond, one_step=True).lower() + '\n\n(s)'
        #else:
            #dnf_cond, untill = split_dnf
            #until, unless = untill.split(' UNLESS ')
            #until_cond = 'until ' + pred2expr(until, until=True) if 'IT' not in until \
            #             else "as long as possible"
            #txt += dnf2command(dnf_cond) + ". "
            #txt += "\n\nRepeat this step " + until_cond + " unless " + \
            #       pred2expr(unless, until=True)+' -- then stop at the previous step.\n\n(s)'
        id_dict[dnf_cond] = i    
            
    if go_to != None:
        goto_split = go_to.split(' UNLESS ')
        if len(goto_split) == 2:
            go_to, unless = goto_split
        else:
            go_to, unless = goto_split[0], None
        txt += str(len(dnfs)+1) + ". GOTO step " + str(id_dict[go_to]+1)
        if unless != None:
            txt += ' unless ' + pred2expr(unless, until=True)
        txt += '.'
    print("\n\n        Whole instructions: \n\n{}".format(txt))
    return txt
    
def alternative_procedures2text(procedures_str):
    """
    Separate alternative procedural formulas and create a natural language
    descriptions for each.
    """
    procs = procedures_str.split('\n\nOR\n\n')
    out = ''
    for n, p in enumerate(procs):
        if n>0: out += '\n\nOR\n\n'
        out += aligned_print(procedure2text(p))
    print("\n\n\n          FINAL INSTRUCTIONS: \n\n{}".format(out))
    return out
    
def aligned_print(txt):
    """
    Format text so that it took 80 lines and then new line was started.
    
    Changed to fit the particular description and some tabulation was added
    for arrows ->, etc.
    """
    new_txt = ''
    steps = txt.split('\n\n(s)')
    counter = 80
    for step in steps:
        counter = 80
        arrow = False
        for num, s in enumerate(step):
            new_txt += s
            if num+1 < len(step) and s+step[num+1] =='\n ':
                arrow = True
                new_txt += '  '
                counter = 78
            elif s == '\n':
                arrow = False
                new_txt += '   '
                counter = 77
            counter -= 1
            if counter <= 0 and (num+1 >= len(step) or step[num+1] == ' ') and arrow:
                new_txt += '\n     '
                counter = 74
            elif counter <= 0 and (num+1 >= len(step) or step[num+1] == ' '):
                new_txt += '\n  '
                counter = 77
        new_txt += '\n\n'
    return(new_txt)
