package it.units.malelab.jgea.core.order;

import it.units.malelab.jgea.core.Individual;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.order.PartiallyOrderedCollection;
import it.units.malelab.jgea.core.util.ArrayTable;
import it.units.malelab.jgea.core.util.TableMap;
import it.units.malelab.jgea.core.util.Pair;
import org.apache.commons.math3.random.AbstractRandomGenerator;
import org.checkerframework.checker.units.qual.A;

/**
 * @author andrea
 */
public class MapElites<T> implements PartiallyOrderedCollection<T> {

    protected final HashMap<List<Integer>, T> archive;
    protected final Boolean maximize;
    protected final Function<T, List<Double>> descriptor;
    protected final Function<T, Double> helper;
    protected final PartialComparator<? super T> comparator;
    protected final List<Double> threshold;
    protected final List<Integer> size;
    protected final List<Double> min;
    protected final List<Double> max;
    public ArrayList<T> lastAddedPerformance;
    public int counter = 0;
    public int counter2 = 0;

    public MapElites(List<Integer> size, List<Double> min, List<Double> max, Boolean maximize, Function<T, List<Double>> descriptor, PartialComparator<? super T> comparator, Function<T, Double> helper) {
        archive = new HashMap<>();
        this.maximize = maximize;
        this.descriptor = descriptor;
        this.helper = helper;
        this.comparator = comparator;
        this.min = min;
        this.max = max;
        this.threshold = new ArrayList<>();
        this.size = size;
        for (int i = 0; i < size.size(); i++) {
            threshold.add((max.get(i) - min.get(i)) / size.get(i));
        }
    }

    public ArrayList<Integer> calcIndexes(List<Double> indexes) {
        ArrayList<Integer> newIndexes = new ArrayList<>();
        //System.out.println("-------");
        for (int i = 0; i < threshold.size(); i++) {
            //System.out.println(i+" "+threshold.get(i)+" "+max.get(i)+" "+min.get(i));
            Integer index = (int) (indexes.get(i) / threshold.get(i));
            if (indexes.get(i) < min.get(i)) {
                newIndexes.add(0);
            } else if (indexes.get(i) > max.get(i)) {
                newIndexes.add(size.get(i) - 1);
            } else {
                newIndexes.add(index);
            }

            //System.out.println(indexes.get(i)+"   "+index+"   "+threshold.get(i)+"   "+ newIndexes.get(i));

        }
        //System.out.println("----------");
        Integer tmp = newIndexes.get(1);
        newIndexes.set(1,newIndexes.get(2));
        newIndexes.set(2,tmp);
        //System.out.println("-------");
        return newIndexes;
    }

    public ArrayList<Integer> index(T individual) {
        List<Double> indexes = this.descriptor.apply(individual);
        return calcIndexes(indexes);
    }

    public boolean shouldBeRecorded(T individual) {
        List<Double> indexes = this.descriptor.apply(individual);
        T oldInd = archive.get(calcIndexes(indexes));

        if (oldInd == null) {
            //System.out.println(calcIndexes(indexes).toString());
            return true;
        } else {
            if (maximize) {

                //System.out.println(calcIndexes(indexes).toString()+";"+helper.apply(individual)+";"+helper.apply(oldInd));
                if (comparator.compare(individual, oldInd).equals(PartialComparator.PartialComparatorOutcome.AFTER)) {
                    counter2 += 1;
                    //System.out.println(calcIndexes(indexes).toString());
                    return true;
                }
            } else {
                if (comparator.compare(individual, oldInd).equals(PartialComparator.PartialComparatorOutcome.BEFORE)) {
                    counter2 += 1;
                    return true;
                }
            }
        }

        counter += 1;
        return false;
    }

    public double getCrowedness(T individual){
        List<Integer> indexes = index(individual);
        int crowed =0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    for (int l = -1; l <= 1; l++) {
                        if (i != 0 && j != 0 && k != 0 && l != 0) {
                            List<Integer> ni = indexes;
                            ni.set(0,ni.get(0)+i);
                            ni.set(1,ni.get(1)+j);
                            ni.set(2,ni.get(2)+k);
                            ni.set(3,ni.get(3)+l);
                            crowed += archive.containsKey(ni)? 1:0;
                        }
                    }
                 }
            }

        }
        return 1 - crowed/(Math.pow(3,4)-1);

    }

    @Override
    public Collection<T> all() {
        return Collections.unmodifiableCollection(archive.values());
    }

    @Override
    public Collection<T> firsts() {
        return all();
    }

    @Override
    public Collection<T> lasts() {
        return all();
    }

    @Override
    public boolean remove(T t) {
        List<Double> indexes = this.descriptor.apply(t);
        T oldInd = archive.get(calcIndexes(indexes));
        if (oldInd.equals(t)) {
            archive.put(calcIndexes(indexes), null);
            return true;
        }

        return false;
    }

    public void add(T individual) {
        List<Integer> convertedIndexes = index(individual);
        T oldInd = archive.get(convertedIndexes);

        if (oldInd != null) {
            if (maximize) {
                if (helper.apply(individual) >= helper.apply(oldInd)) {

                    archive.put(convertedIndexes, individual);
                    this.lastAddedPerformance.add(individual);
                }
            } else {
                if (helper.apply(individual) <= helper.apply(oldInd)) {
                    archive.put(convertedIndexes, individual);
                    this.lastAddedPerformance.add(individual);
                }
            }
        } else {
            archive.put(convertedIndexes, individual);
            this.lastAddedPerformance.add(individual);
        }
    }


    public void addAll(Collection<T> individuals) {
        this.lastAddedPerformance = new ArrayList<>();
        for (T individual : individuals) {
            this.add(individual);
        }
    }


    public Collection<T> values() {
        Collection<T> pops = archive.values();
        return pops;
    }


}



