package it.units.malelab.jgea.core.evolver;

import com.google.common.base.Stopwatch;
import it.units.malelab.jgea.core.Factory;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.core.order.*;
import org.checkerframework.common.value.qual.IntRange;

import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class MapElitesWithEvolvabilityEvolver<S,F> extends AbstractIterativeEvolver<List<Double>,S,F> {
    public enum SelectionStrategies {UNIFORM, CROWDEDNESS}

    protected static final Logger L = Logger.getLogger(MapElitesEvolver.class.getName());
    protected MapElitesWithEvolvability<Individual<List<Double>,S,F>> population;
    protected Mutation<List<Double>> mutation;
    protected int populationSize;
    protected int batch_size;
    protected SelectionStrategies selectionStrategies;

    public MapElitesWithEvolvabilityEvolver(Function<Individual< List<Double>,S,F>, List<Double>> descriptor,
                                     List<Double> max,
                                     List<Double> min,
                                     List<Integer> size,
                                     Function<List<Double>, ? extends S> solutionMapper,
                                     Factory<List<Double>> genotypeFactory,
                                     PartialComparator<? super Individual<List<Double>, S, F>> individualComparator,
                                     Mutation<List<Double>> mutation, int populationSize, int batch_size, Function<Individual< List<Double>,S,F>, Double> helper,
                                            BiFunction<Individual< List<Double>,S,F>, Integer, double[]> setEvo, SelectionStrategies se) {
        super(solutionMapper, genotypeFactory, individualComparator);
        population = new MapElitesWithEvolvability<>(size, min, max, true, descriptor, individualComparator, helper, setEvo);
        this.mutation = mutation;
        this.populationSize = populationSize;
        this.batch_size = batch_size;
        this.selectionStrategies = se;
    }

    @Override
    public Collection<S> solve(Function<S, F> fitnessFunction, Predicate<? super Event<List<Double>, S, F>> stopCondition, Random random, ExecutorService executor, Listener<? super Event<List<Double>, S, F>> listener) throws InterruptedException, ExecutionException {
        State state = initState();
        Stopwatch stopwatch = Stopwatch.createStarted();
        Collection<Individual<List<Double>, S, F>> newPops = initPopulation(fitnessFunction, random, executor, state);
        List<List<Individual<List<Double>, S, F>>> children = buildChildren(newPops, fitnessFunction, random, executor, state);

        population.addAll(new ArrayList<>(newPops), children);

        while (true) {

            state.setElapsedMillis(stopwatch.elapsed(TimeUnit.MILLISECONDS));

            Event<List<Double>, S, F> event = new Event<>(state, population,
                    new DAGPartiallyOrderedCollection<>(population.lastAddedPerformance, individualComparator),
                    new DAGPartiallyOrderedCollection<>(population.lastAddedEvolvability, individualComparator));

            listener.listen(event);
            if (stopCondition.test(event)) {
                System.out.println(population.values().size()+" not recorded "+ population.counter+" updated "+population.counter2);
                L.fine(String.format("Stop condition met: %s", stopCondition.toString()));
                break;
            }

            newPops = updatePopulation(population, fitnessFunction, random, executor, state);
            children = buildChildren(newPops, fitnessFunction, random, executor, state);
            population.addAll(new ArrayList<>(newPops), children);
            L.fine(String.format("Population updated: %d individuals", population.size()));
            state.incIterations(1);
        }
        listener.done();
        return new DAGPartiallyOrderedCollection<>(population.values(), individualComparator).firsts().stream()
                .map(Individual::getSolution)
                .collect(Collectors.toList());
    }

    @Override
    protected Collection<Individual<List<Double>, S, F>> initPopulation(Function<S, F> fitnessFunction, Random random, ExecutorService executor, State state) throws ExecutionException, InterruptedException {
        return initPopulation(populationSize, fitnessFunction, random, executor, state);
    }

    @Override
    protected Collection<Individual<List<Double>, S, F>> updatePopulation(PartiallyOrderedCollection<Individual<List<Double>, S, F>> orderedPopulation, Function<S, F> fitnessFunction, Random random, ExecutorService executor, State state) throws ExecutionException, InterruptedException {
        Collection<Individual<List<Double>, S, F>> offspring = buildOffspring(orderedPopulation, new DAGPartiallyOrderedCollection<>(population.getEvolvabilityPopulation(), individualComparator), fitnessFunction, random, executor, state);
        return offspring;
    }


    protected Collection<Individual<List<Double>, S, F>> buildOffspring(PartiallyOrderedCollection<Individual<List<Double>, S, F>> orderedPopulation, PartiallyOrderedCollection<Individual<List<Double>, S, F>> orderedPopulation1, Function<S, F> fitnessFunction, Random random, ExecutorService executor, State state) throws ExecutionException, InterruptedException {
        List<Individual<List<Double>,S,F>> allGenotypesPerformance = orderedPopulation.all().stream().filter(Objects::nonNull).collect(Collectors.toList());
        List<Individual<List<Double>,S,F>> allGenotypesEvolvability = orderedPopulation1.all().stream().filter(Objects::nonNull).collect(Collectors.toList());
        DAGPartiallyOrderedCollection<Individual<List<Double>,S,F>> offspring = new DAGPartiallyOrderedCollection<>(individualComparator);

        if(this.selectionStrategies == SelectionStrategies.UNIFORM){
            offspring = uniformSelection(allGenotypesPerformance,allGenotypesEvolvability, offspring, random);
        }else{
            offspring = rouletteWheelSelection(allGenotypesPerformance,allGenotypesEvolvability, offspring, random);
        }


        Collection<List<Double>> offspringGenotypes = offspring.all().stream()
                .map(i -> mutation.mutate(i.getGenotype(), random))
                .collect(Collectors.toList());
        return AbstractIterativeEvolver.map(offspringGenotypes, List.of(), solutionMapper, fitnessFunction, executor, state);
    }

    protected List<List<Individual<List<Double>, S, F>>> buildChildren(Collection<Individual<List<Double>, S, F>> population, Function<S, F> fitnessFunction, Random random, ExecutorService executor, State state) throws ExecutionException, InterruptedException{
        ArrayList<List<Individual<List<Double>, S, F>>> children = new ArrayList<>();

        for (Individual<List<Double>, S, F> ind : population ){
            Collection<List<Double>> offspringGenotypes =IntStream.range(0, batch_size).mapToObj(i -> mutation.mutate(ind.getGenotype(), random)).collect(Collectors.toList());
            children.add(new ArrayList<>(AbstractIterativeEvolver.map(offspringGenotypes, List.of(), solutionMapper, fitnessFunction, executor, state)));

        }
        return children;

    }
    protected DAGPartiallyOrderedCollection<Individual<List<Double>,S,F>> uniformSelection(List<Individual<List<Double>,S,F>> allGenotypesPerformance, List<Individual<List<Double>,S,F>> allGenotypesEvolvability, DAGPartiallyOrderedCollection<Individual<List<Double>,S,F>> offspring, Random random){
        for (int c = 0; c< (batch_size/2); c++) {
            offspring.add(allGenotypesPerformance.get(random.nextInt(allGenotypesPerformance.size())));
        }
        for (int c = 0; c< batch_size-(batch_size/2); c++) {
            offspring.add(allGenotypesEvolvability.get(random.nextInt(allGenotypesEvolvability.size())));
        }
        return offspring;
    }

        protected DAGPartiallyOrderedCollection<Individual<List<Double>,S,F>> rouletteWheelSelection(List<Individual<List<Double>,S,F>> allGenotypesPerformance, List<Individual<List<Double>,S,F>> allGenotypesEvolvability, DAGPartiallyOrderedCollection<Individual<List<Double>,S,F>> offspring, Random random){
        double[] pDistribution =new double[allGenotypesPerformance.size()];
        double[] eDistribution =new double[allGenotypesEvolvability.size()];

        double psum = 0;
        double esum = 0;
        for (int i = 0; i < pDistribution.length; i++) {
            pDistribution[i] = population.getCrowedness(allGenotypesPerformance.get(i));
            psum += pDistribution[i];
        }

        double finalPsum = psum;
        double[] finalPDistribution = pDistribution;
        pDistribution = IntStream.range(1,pDistribution.length).mapToDouble(i -> i==0? (finalPDistribution[i]/ finalPsum): finalPDistribution[i-1]+(finalPDistribution[i]/ finalPsum)).toArray();

        for (int i = 0; i < eDistribution.length; i++) {
            eDistribution[i] = population.getCrowednessEvolvability(allGenotypesEvolvability.get(i));
            esum += eDistribution[i];
        }

        double finalEsum =esum;
        double[] finalEDistribution = eDistribution;
        eDistribution = IntStream.range(1,eDistribution.length).mapToDouble(i -> i==0? (finalEDistribution[i]/ finalEsum): finalEDistribution[i-1]+(finalEDistribution[i]/ finalEsum)).toArray();



        for (int c = 0; c< (batch_size/2); c++) {
            double rnd = random.nextDouble();
            boolean flag =true;
            for (int i = 0; i < finalPDistribution.length && flag; i++) {
                if(rnd < finalEDistribution[i]){
                    flag= false;
                    offspring.add(allGenotypesPerformance.get(i));
                }
            }
        }
        for (int c = 0; c< batch_size -(batch_size/2); c++) {
            double rnd = random.nextDouble();
            boolean flag =true;
            for (int i = 0; i < finalEDistribution.length && flag; i++) {
                if(rnd < finalEDistribution[i]){
                    flag= false;
                    offspring.add(allGenotypesEvolvability.get(i));
                }
            }
        }
        return offspring;
    }
}
