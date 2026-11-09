"""Runs only inside Bubblewrap with SOURCE, GENERATED, MUTANTS, FUT supplied."""
import ast
import json

valid=[]
rounds=[]
for snippet in GENERATED:
    entry={'submitted':0,'valid':0,'errors':[]}
    try:
        tree=ast.parse(snippet)
        if any(isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) for node in tree.body):
            raise ValueError('assert-only interface forbids redefining functions/classes')
        imports='\n'.join(ast.unparse(node) for node in tree.body if isinstance(node,(ast.Import,ast.ImportFrom)))
        assertions=[node for node in tree.body if isinstance(node,ast.Assert)]
        if not assertions:
            raise ValueError('no top-level assertions')
        entry['submitted']=min(len(assertions),8)
        for node in assertions[:8]:
            invokes=any(isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id==FUT
                        for call in ast.walk(node.test))
            if not invokes:
                entry['errors'].append('assert does not call FUT')
                continue
            test=imports+'\n'+ast.unparse(node)
            namespace={'__name__':'candidate'}
            try:
                exec(compile(SOURCE,'candidate.py','exec'),namespace)
                exec(compile(test,'generated_test.py','exec'),namespace)
                valid.append(test)
                entry['valid']+=1
            except BaseException as error:
                entry['errors'].append(type(error).__name__)
    except BaseException as error:
        entry['submitted']=max(1,entry['submitted'])
        entry['errors'].append(type(error).__name__+': '+str(error))
    rounds.append(entry)

killed=[]
for mutant in MUTANTS:
    failed=False
    for test in valid:
        namespace={'__name__':'candidate'}
        try:
            exec(compile(mutant,'mutant.py','exec'),namespace)
            exec(compile(test,'generated_test.py','exec'),namespace)
        except BaseException:
            failed=True
            break
    killed.append(failed)

# PROFILE_SOURCE collects coverage of the same valid tests on the reference FUT.
TESTS=valid or ['pass']
exec(compile(PROFILE_SOURCE,'profile_driver.py','exec'),globals())
coverage_result=result
if not valid:
    coverage_result['branch_covered']=0
    coverage_result['line_covered']=0
metrics={'rounds':rounds,'submitted_tests':sum(r['submitted'] for r in rounds),
         'valid_tests':len(valid),'mutants':len(MUTANTS),'killed_mutants':sum(killed),
         'mutant_kills':killed,'coverage':{k:coverage_result.get(k) for k in
                                         ('branch_total','branch_covered','line_total','line_covered','error')}}
print('__A1_ULT__'+json.dumps(metrics))
